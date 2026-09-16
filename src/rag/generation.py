"""
generation.py

Utilitas untuk memuat model generator RAG hasil fine-tuning sendiri,
membangun pipeline text generation, memformat konteks retrieval, dan
menyusun prompt runnable untuk pipeline RAG (Legal AI Assistant).
"""

import torch
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

SYSTEM_PROMPT_RAG = (
    "Kamu adalah asisten AI Legal Team yang menjawab pertanyaan hukum "
    "HANYA berdasarkan konteks dokumen yang diberikan. "
    "Jika jawaban tidak ditemukan dalam konteks, katakan dengan jujur bahwa "
    "kamu tidak menemukan informasinya di dokumen -- jangan berspekulasi atau "
    "menambahkan pengetahuan di luar konteks. "
    "Jawab dalam Bahasa Indonesia, singkat dan jelas, dan sebutkan sumber "
    "(nama dokumen/halaman) yang mendasari jawabanmu kalau relevan."
)


def load_finetuned_model(
    hf_repo_id: str,
    load_in_4bit: bool = True,
    hf_token: str | None = None,
):
    """Memuat model generator RAG hasil fine-tuning sendiri dari HF Hub.

    `hf_repo_id` harus menunjuk ke repo hasil `push_to_hub_merged` milik
    sendiri (contoh: `"username/sft-qwen25-3b-run2"`), bukan model dasar
    dari penyedia lain -- lihat `link_huggingface.txt` untuk repo yang
    valid dipakai.

    Quantization 4-bit dipakai juga untuk inference (bukan hanya
    training) agar muat nyaman di GPU bersama embedding model dan
    reranker yang berjalan pada sesi yang sama. Apabila GPU memory
    ketat, ini parameter pertama yang disarankan dilonggarkan.

    Catatan compute dtype: `bnb_4bit_compute_dtype=torch.bfloat16`
    sudah divalidasi berjalan tanpa error pada GPU T4 (NB04, end-to-end).
    T4 (arsitektur Turing) tidak memiliki tensor core native untuk
    bf16 seperti GPU Ampere ke atas, sehingga secara teori bisa lebih
    lambat dibanding float16 -- namun karena run aktual di T4 sudah
    berhasil tanpa masalah, nilai ini dipertahankan apa adanya. Apabila
    compute dipindah ke GPU lain (terutama P100/Pascal, yang dukungan
    fp16/bf16-nya lebih terbatas dari T4), nilai ini perlu diuji ulang
    atau diganti deteksi otomatis via `torch.cuda.is_bf16_supported()`.

    Args:
        hf_repo_id: Repo HuggingFace Hub hasil fine-tuning sendiri.
        load_in_4bit: Jika True, model dimuat dalam quantized 4-bit
            (nf4, double quantization) untuk inference.
        hf_token: Token HuggingFace, dibutuhkan apabila repo bersifat
            private.

    Returns:
        Tuple `(model, tokenizer)` siap dipakai untuk text generation.

    Raises:
        ValueError: Apabila tokenizer dari `hf_repo_id` tidak memiliki
            `chat_template` terpasang. Divalidasi sebelum model
            dialokasikan ke GPU, supaya kegagalan ini tidak
            membuang GPU memory untuk model yang gagal dipakai.
    """
    tokenizer = AutoTokenizer.from_pretrained(hf_repo_id, token=hf_token)

    if tokenizer.chat_template is None:
        raise ValueError(
            f"Tokenizer dari {hf_repo_id} tidak punya chat_template. "
            f"Ini harusnya ter-carry otomatis dari push_to_hub_merged "
            f"saat training (chat template 'qwen-2.5'). Cek ulang "
            f"proses push."
        )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        hf_repo_id,
        quantization_config=bnb_config if load_in_4bit else None,
        device_map="auto",
        token=hf_token,
    )

    return model, tokenizer


def build_text_generation_pipeline(
    model,
    tokenizer,
    max_new_tokens: int = 1000,
    temperature: float = 0.2,
) -> HuggingFacePipeline:
    """Membangun pipeline text generation untuk jawaban akhir RAG.

    `do_sample=True` dengan `temperature` rendah (0.2) dipilih untuk
    variasi minimal namun tetap konservatif -- cocok untuk legal QA
    yang membutuhkan presisi.

    Alternatif `do_sample=False` (greedy) lebih deterministic dan
    reproducible, yang untuk domain legal sebenarnya juga defensible
    (jawaban idealnya tidak "berubah-ubah" untuk pertanyaan yang sama).
    Tetap dipilih `do_sample=True` untuk konsisten dengan pola awal
    project -- apabila pengujian RAG menunjukkan jawaban goyang antar
    run untuk query yang identik, ini parameter pertama yang
    disarankan diganti ke `do_sample=False`.

    Args:
        model: Model generator hasil `load_finetuned_model`.
        tokenizer: Tokenizer yang berpasangan dengan `model`.
        max_new_tokens: Jumlah token maksimum yang dihasilkan.
        temperature: Temperature sampling, rendah untuk jawaban yang
            lebih konservatif.

    Returns:
        `HuggingFacePipeline` siap dipakai sebagai LLM di RAG chain.
    """
    text_gen_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=temperature,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=max_new_tokens,
    )
    return HuggingFacePipeline(pipeline=text_gen_pipeline)


def format_context(docs: list[Document]) -> str:
    """Menggabungkan Document hasil retrieval menjadi satu string konteks.

    Tiap chunk diberi label sumber di awal, sehingga model bisa "melihat"
    dari dokumen/halaman/UU mana suatu potongan konteks berasal --
    menjadi fondasi untuk kemampuan sitasi jawaban (menyebut sumber
    pasal/UU), bukan sekadar teks polos tanpa atribusi.

    Document hasil web fallback (`source_type="web"`) tidak memiliki
    `uu_number`/`pasal_refs` seperti dokumen lokal, sehingga diberi
    format sitasi berbeda: judul dan url, bukan nomor UU/pasal.

    Args:
        docs: List Document hasil retrieval (lokal maupun web fallback).

    Returns:
        String gabungan seluruh chunk dengan label sumber per chunk,
        dipisah baris kosong ganda.
    """
    parts = []
    for doc in docs:
        if doc.metadata.get("source_type") == "web":
            header = f"[Sumber web: {doc.metadata.get('title', '?')} ({doc.metadata.get('url', '?')})]"
        else:
            header = (
                f"[{doc.metadata.get('uu_number', '?')}, "
                f"{doc.metadata.get('source_file', '?')}]"
            )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


def build_prompt_runnable(
    tokenizer, system_prompt: str = SYSTEM_PROMPT_RAG
) -> RunnableLambda:
    """Membangun Runnable yang menyusun prompt via chat template tokenizer.

    Menerima dict `{"context": str, "question": str}` dan menghasilkan
    prompt string melalui `tokenizer.apply_chat_template()`. Pendekatan
    ini dipilih (bukan menyusun string prompt dengan token khusus yang
    di-hardcode, misal token format Llama-3) supaya format token selalu
    mengikuti tokenizer model yang sebenarnya dipakai -- format ChatML
    untuk Qwen2.5, atau format lain secara otomatis apabila model dasar
    diganti. Ini menghindari risiko mismatch antara format token yang
    diasumsikan di kode dengan format yang benar-benar diharapkan model.

    Args:
        tokenizer: Tokenizer dengan chat template terpasang.
        system_prompt: System prompt yang disisipkan ke conversation.
            Default `SYSTEM_PROMPT_RAG`.

    Returns:
        `RunnableLambda` yang menerima dict `{"context", "question"}`
        dan mengembalikan prompt string siap di-generate.
    """

    def _format(inputs: dict) -> str:
        context = inputs["context"]
        question = inputs["question"]
        user_content = f"Konteks:\n{context}\n\nPertanyaan: {question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return RunnableLambda(_format)
