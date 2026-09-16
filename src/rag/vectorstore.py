"""
vectorstore.py

Utilitas untuk membangun embedding model dan vector store (ChromaDB)
pada pipeline RAG (Legal AI Assistant).
"""

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_PERSIST_DIR = "/kaggle/working/chroma_db"


def build_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> HuggingFaceEmbeddings:
    """Memuat embedding model open-source (bge-m3), dengan auto-detect device.

    Bagian yang perlu diperhatikan: bge-m3 dilatih dengan cosine
    similarity sebagai objective, sehingga embedding yang dihasilkan
    harus dinormalisasi (L2 norm = 1) agar cosine similarity berperilaku
    benar. Apabila tidak dinormalisasi, hasil similarity antar chunk
    bisa bias ke chunk yang magnitude vektornya kebetulan lebih besar,
    bukan yang paling relevan secara makna.

    Args:
        model_name: Nama model embedding di HuggingFace Hub.

    Returns:
        Instance `HuggingFaceEmbeddings` dengan `normalize_embeddings=True`,
        dijalankan di GPU apabila tersedia, jika tidak di CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "[WARNING] CUDA tidak terdeteksi, embedding akan jalan di CPU "
            "(jauh lebih lambat untuk 4 dokumen UU + child chunking)."
        )

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(
    embedding_model: HuggingFaceEmbeddings,
    collection_name: str = "legal_docs",
    persist_directory: str = DEFAULT_PERSIST_DIR,
) -> Chroma:
    """Membuat Chroma collection kosong, siap diisi lewat retriever.

    Collection dibuat kosong -- pengisian (splitting + `add_documents`)
    ditangani oleh `ParentDocumentRetriever` di modul `retrievers.py`,
    bukan di sini.

    Bagian yang perlu diperhatikan: default distance metric Chroma
    adalah L2, BUKAN cosine. Karena `embedding_model` yang dipakai sudah
    dinormalisasi untuk cosine similarity, collection ini harus dipaksa
    memakai `'hnsw:space': 'cosine'` juga -- apabila tidak, akan terjadi
    mismatch antara asumsi training embedding (cosine) dan metric yang
    dipakai saat search (L2), yang membuat hasil retrieval sedikit
    menyimpang dari yang seharusnya.

    `persist_directory` default ke `/kaggle/working` supaya collection
    tetap ada apabila kernel restart dalam sesi yang sama (masuk tab
    Output Kaggle). Folder ini TIDAK boleh ikut di-commit ke GitHub --
    isinya binary SQLite dan index berukuran besar, dan dapat dibangun
    ulang kapan saja dari 4 PDF sumber. Pastikan `chroma_db/` sudah
    masuk `.gitignore` sebelum push.

    Args:
        embedding_model: Instance embedding model hasil
            `build_embedding_model`.
        collection_name: Nama collection Chroma.
        persist_directory: Path direktori penyimpanan persist Chroma.

    Returns:
        Instance `Chroma` kosong, siap diisi lewat
        `ParentDocumentRetriever`.
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
