"""
hyde.py

Utilitas HyDE (Hypothetical Document Embeddings) untuk query
transformation pada pipeline RAG (Legal AI Assistant). Menghasilkan
jawaban hipotesis dari LLM untuk dipakai sebagai query retrieval
tambahan, alih-alih hanya query asli pengguna.
"""

import re

from langchain_core.documents import Document
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline as hf_pipeline

from src.rag.retrievers import dedup_documents

THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

HYDE_SYSTEM_PROMPT = (
    "Kamu adalah asisten hukum yang menjawab pertanyaan secara singkat "
    "dan percaya diri, seolah kamu sudah tahu jawabannya berdasarkan "
    "peraturan yang relevan. Jawaban ini HANYA dipakai untuk membantu "
    "proses pencarian dokumen, BUKAN jawaban final ke pengguna -- jawab "
    "singkat (2-4 kalimat), boleh menyebut istilah/peraturan meskipun "
    "kamu tidak yakin persis nomornya."
)


def _strip_think_block(text: str) -> str:
    """Membuang blok `<think>...</think>` dari hypothesis HyDE.

    Isi reasoning secara semantik berbeda dari jawaban legal itu
    sendiri, sehingga berisiko men-dilute similarity ke chunk yang
    relevan apabila ikut dipakai untuk retrieval embedding.

    Args:
        text: Teks hypothesis mentah dari hasil generation.

    Returns:
        Teks tanpa blok `<think>...</think>`, sudah di-strip whitespace.
    """
    return THINK_BLOCK_PATTERN.sub("", text).strip()


def build_hyde_pipeline(
    llm: HuggingFacePipeline,
    max_new_tokens: int = 200,
    temperature: float = 0.9,
) -> HuggingFacePipeline:
    """Membangun pipeline generation kedua khusus untuk HyDE.

    Me-reuse model dan tokenizer persis sama dari `llm` generator RAG
    utama (`llm.pipeline.model` / `.tokenizer`) -- tidak memuat ulang
    dari HF Hub, sehingga tidak menambah GPU memory sama sekali.

    `max_new_tokens` dibuat lebih pendek (200, dibanding 1000 pada
    generator utama) dan `temperature` lebih tinggi (0.9, dibanding
    0.2) karena HyDE membutuhkan hypothesis yang singkat dan bervariasi
    antar generation -- kebutuhan yang berbeda dari jawaban final yang
    harus presisi dan konservatif.

    Args:
        llm: `HuggingFacePipeline` generator RAG utama, sumber model
            dan tokenizer yang di-reuse.
        max_new_tokens: Jumlah token maksimum untuk tiap hypothesis.
        temperature: Temperature sampling, tinggi untuk variasi antar
            hypothesis.

    Returns:
        `HuggingFacePipeline` baru yang berbagi model dan tokenizer
        dengan `llm`, dengan konfigurasi generation terpisah.
    """
    base_pipeline = llm.pipeline
    hyde_hf_pipeline = hf_pipeline(
        model=base_pipeline.model,
        tokenizer=base_pipeline.tokenizer,
        task="text-generation",
        temperature=temperature,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=max_new_tokens,
    )
    return HuggingFacePipeline(pipeline=hyde_hf_pipeline)


def generate_hypothetical_answers(
    query: str,
    hyde_llm: HuggingFacePipeline,
    tokenizer,
    n: int = 2,
) -> list[str]:
    """Menghasilkan n jawaban hipotesis untuk query (HyDE query transformation).

    `hyde_llm` harus berupa instance hasil `build_hyde_pipeline` --
    me-reuse model yang sama dengan generator RAG utama, bukan memuat
    model baru.

    Known limitation: fungsi ini tidak melakukan filtering terhadap
    hypothesis kosong. Apabila `_strip_think_block` menghasilkan string
    kosong (misal model hanya mengeluarkan blok `<think>` tanpa jawaban
    di luar itu), string kosong tersebut tetap dimasukkan ke list hasil
    dan akan diteruskan sebagai query ke retriever oleh pemanggil
    (`hyde_retrieve`) -- perilaku retriever untuk query kosong
    bergantung pada backend yang dipakai (BM25/dense) dan tidak
    dijamin konsisten. Secara praktis risiko ini kecil karena
    `HYDE_SYSTEM_PROMPT` secara eksplisit meminta jawaban singkat namun
    tetap berisi, bukan kosong.

    Catatan performa: generation dilakukan `n` kali secara berurutan
    lewat `hyde_llm.invoke()`, masing-masing memproses ulang prompt
    dari awal tanpa berbagi komputasi antar call. Untuk `n` kecil
    (sesuai kebutuhan HyDE pada umumnya, minimal 2) overhead ini
    minor, namun akan semakin tidak efisien apabila `n` dinaikkan
    signifikan -- pada kondisi tersebut, generation batched sekaligus
    lebih disarankan dibanding loop ini.

    Args:
        query: Query asli dari pengguna.
        hyde_llm: Pipeline HyDE hasil `build_hyde_pipeline`.
        tokenizer: Tokenizer dengan chat template terpasang, dipakai
            untuk membangun prompt HyDE.
        n: Jumlah hypothesis yang dihasilkan.

    Returns:
        List string hypothesis sepanjang `n`, blok `<think>` sudah
        dibuang dari masing-masing.
    """
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": HYDE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    hypotheses = []
    for _ in range(n):
        raw_output = hyde_llm.invoke(prompt)
        hypotheses.append(_strip_think_block(raw_output))
    return hypotheses


def hyde_retrieve(
    query: str,
    hyde_answers: list[str],
    retriever,
    top_k: int = 10,
) -> list[Document]:
    """Melakukan retrieval gabungan query asli dan hypothesis HyDE.

    Dipakai untuk demonstrasi standalone di notebook (bukan dipakai
    secara internal oleh `RAGPipeline`, yang sudah menangani union
    HyDE sendiri lewat `_retrieve_with_hyde`). Retrieval dilakukan
    memakai query asli ditambah tiap hypothetical answer, hasilnya
    di-union dan di-dedup.

    Args:
        query: Query asli dari pengguna.
        hyde_answers: List hypothesis hasil `generate_hypothetical_answers`.
        retriever: Retriever yang dipanggil untuk tiap query (query
            asli maupun tiap hypothesis).
        top_k: Jumlah dokumen maksimum yang dikembalikan setelah dedup.

    Returns:
        List `Document` hasil union dan dedup, dipotong sampai `top_k`.
    """
    doc_lists = [retriever.invoke(query)] + [
        retriever.invoke(ans) for ans in hyde_answers
    ]
    combined = dedup_documents(doc_lists)
    return combined[:top_k]
