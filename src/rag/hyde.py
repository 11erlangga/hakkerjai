import re
from typing import Optional

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
    """
    Buang blok <think>...</think> sebelum hypothesis dipakai untuk
    retrieval embedding -- isi reasoning secara semantic beda dari jawaban
    legal itu sendiri, berisiko men-dilute similarity ke chunk relevan.
    """
    return THINK_BLOCK_PATTERN.sub("", text).strip()


def build_hyde_pipeline(
    llm: HuggingFacePipeline,
    max_new_tokens: int = 200,
    temperature: float = 0.9,
) -> HuggingFacePipeline:
    """
    Bikin pipeline generation KEDUA khusus HyDE, reuse model+tokenizer
    PERSIS SAMA dari llm generator RAG utama (llm.pipeline.model/.tokenizer)
    -- TIDAK load ulang dari HF Hub, NOL tambahan GPU memory.

    max_new_tokens lebih pendek (200 vs 1000 di generator utama) dan
    temperature lebih tinggi (0.9 vs 0.2) -- HyDE butuh hypothesis singkat
    & bervariasi antar generation, beda kebutuhan dari jawaban final yang
    harus presisi/conservative.
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
    """
    Generate n hypothetical answer untuk query (HyDE query transformation).
    hyde_llm HARUS instance dari build_hyde_pipeline() -- reuse model yang
    sama, BUKAN load model baru.
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
    """
    Untuk demonstrasi standalone di notebook (bukan dipakai internal
    RAGPipeline, yang sudah handle union sendiri di _retrieve_with_hyde).
    Retrieve pakai query asli + tiap hypothetical answer, union dedup.
    """
    doc_lists = [retriever.invoke(query)] + [
        retriever.invoke(ans) for ans in hyde_answers
    ]
    combined = dedup_documents(doc_lists)
    return combined[:top_k]
