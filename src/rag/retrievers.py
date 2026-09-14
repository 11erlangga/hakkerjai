from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
    ParentDocumentRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.stores import InMemoryByteStore

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


def build_dense_retriever(
    vectorstore,
    parent_splitter,
    child_splitter,
    search_k: int = 10,
) -> ParentDocumentRetriever:
    """
    Dense retriever: child chunks di-embed & di-search, tapi yang
    dikembalikan ke LLM adalah parent chunk (konteks lebih utuh).

    docstore selalu in-memory (bukan persist_directory seperti Chroma) --
    ini konsekuensi dari desain ParentDocumentRetriever: parent chunk
    disimpan sebagai objek Python biasa (bukan vector), jadi kalau kernel
    Kaggle restart, docstore ini HILANG meskipun vectorstore-nya persist.
    Artinya kamu harus re-run retriever.add_documents(...) tiap sesi baru,
    walau Chroma collection-nya sendiri sudah ke-cache di disk. Ini bukan
    bug, tapi trade-off yang perlu kamu sadar sebelum debug "kok konteks
    kosong padahal vectorstore-nya udah ke-load".
    """
    docstore = InMemoryByteStore()
    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_type="similarity",
        search_kwargs={"k": search_k},
    )


def ingest_into_dense_retriever(
    retriever: ParentDocumentRetriever,
    documents: list[Document],
    batch_size: int = 50,
) -> None:
    """
    Trigger parent+child splitting internal ParentDocumentRetriever, isi
    vectorstore (child) + docstore (parent) sekaligus.
    Panggil ini SEKALI per sesi kernel setelah retriever dibuat.

    FIX: dokumen di-push per BATCH (bukan semua sekaligus), karena Chroma
    membatasi jumlah embedding yang bisa di-upsert dalam satu panggilan API
    (di versi yang kita pakai: max 5461). Kalau semua parent documents
    (ratusan halaman dari 4 PDF UU) di-push sekaligus, hasil child chunks-nya
    bisa jauh melebihi limit itu -- kejadian di kita: 6583 chunk vs limit 5461.

    batch_size=50 di sini adalah jumlah PARENT-level input documents (halaman
    PDF) per batch, BUKAN jumlah child chunks -- karena kita gak kontrol
    langsung berapa child chunks dihasilkan per halaman (tergantung isi
    halaman, bisa 1 chunk bisa belasan). 50 halaman per batch itu conservative
    margin di bawah limit 5461, cukup aman kecuali halaman-halaman itu jauh
    lebih padat teks dari biasanya -- kalau masih kena limit error lagi,
    turunkan batch_size ini lebih kecil (misal 20).
    """
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        retriever.add_documents(batch)
        print(
            f"Ingested batch {i // batch_size + 1}: {len(batch)} dokumen "
            f"({i + len(batch)}/{len(documents)} total)"
        )


def build_bm25_retriever(
    documents: list[Document],
    child_splitter,
    k: int = 10,
) -> BM25Retriever:
    """
    BM25 index dibangun dari CHILD-sized chunks (bukan raw page-level
    documents seperti versi awal kamu) -- supaya granularitas konsisten
    dengan dense retriever, jadi saat di-ensemble nanti kedua retriever
    "berbicara dalam unit yang sama" (chunk ~400 char), bukan BM25
    mengembalikan 1 halaman penuh sementara dense mengembalikan potongan
    kecil.

    Catatan: chunk ini di-split langsung dari `documents` pakai
    child_splitter, TERPISAH dari child chunks yang dibuat internal oleh
    ParentDocumentRetriever (yang di-split dari parent chunks dulu, baru
    child). Jadi batas chunk-nya tidak dijamin identik character-per-
    character dengan yang ada di dense index -- tapi ukuran & overlap-nya
    sama, yang tadinya jadi concern utama.
    """
    child_chunks = child_splitter.split_documents(documents)
    bm25 = BM25Retriever.from_documents(child_chunks)
    bm25.k = k
    return bm25


def build_ensemble_retriever(
    bm25_retriever: BM25Retriever,
    dense_retriever: ParentDocumentRetriever,
    weights: tuple[float, float] = (0.5, 0.5),
) -> EnsembleRetriever:
    """
    Gabung BM25 (keyword, kuat untuk istilah pasal/nomor UU yang eksak)
    dan dense retriever (semantic, kuat untuk pertanyaan berbahasa natural
    yang gak persis pakai istilah dokumen).

    weights=(0.5, 0.5) itu titik awal netral, BELUM di-tuning. Brief cuma
    minta "tentukan bobot masing-masing" (ada keputusan eksplisit), bukan
    minta hasil optimal -- tapi kalau nanti hasil retrieval kualitatif
    keliatan bias ke salah satu, ini parameter pertama yang diubah.
    """
    return EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=list(weights),
    )


def build_reranked_retriever(
    base_retriever,
    reranker: HuggingFaceCrossEncoder,
    top_n: int = 3,
) -> ContextualCompressionRetriever:
    """
    Cross-encoder reranker: base_retriever (ensemble) ambil kandidat lebih
    banyak dulu (k=10 per retriever), reranker baca ulang tiap kandidat
    BERSAMA query-nya (cross-attention) -- lebih akurat tapi lebih lambat.

    top_n=3 sesuai contoh di brief ("Top-K (misal 3)").
    """
    compressor = CrossEncoderReranker(model=reranker, top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def print_retrieved_docs(docs: list[Document]) -> None:
    """Helper sanity-check: tampilkan chunk + sumber (termasuk UU/pasal) untuk inspeksi manual."""
    for i, doc in enumerate(docs, start=1):
        print(f"--- Dokumen {i} ---")
        print(
            f"Sumber: {doc.metadata.get('source_file', '?')}, "
            f"halaman: {doc.metadata.get('page', '?')}"
        )
        print(
            f"UU/PP: {doc.metadata.get('uu_number', '?')}, "
            f"Pasal terdeteksi: {doc.metadata.get('pasal_refs', []) or '(tidak ada)'}"
        )
        print(doc.page_content)
        print()


def build_reranker(
    reranker_model_name: str = RERANKER_MODEL_NAME,
) -> HuggingFaceCrossEncoder:
    """
    Load cross-encoder reranker SEKALI per sesi kernel. Reuse instance ini
    di build_reranked_retriever() dan di rerank_with_scores() -- jangan
    panggil ulang, supaya gak numpuk 2 instance model reranker di GPU
    memory tanpa perlu.
    """
    return HuggingFaceCrossEncoder(model_name=reranker_model_name)


def dedup_documents(doc_lists: list[list[Document]]) -> list[Document]:
    """
    Gabung beberapa list hasil retrieval jadi satu, dedup berdasar 100
    karakter pertama page_content. Urutan diprioritaskan sesuai urutan
    doc_lists yang dipassing (list pertama menang kalau ada duplikat).
    """
    seen = set()
    combined: list[Document] = []
    for docs in doc_lists:
        for doc in docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                combined.append(doc)
    return combined


def rerank_with_scores(
    reranker: HuggingFaceCrossEncoder,
    query: str,
    documents: list[Document],
) -> list[tuple[Document, float]]:
    """
    Score tiap (query, doc) pair pakai cross-encoder secara manual (akses
    .score() langsung, BUKAN lewat ContextualCompressionRetriever) --
    supaya raw score-nya bisa diakses, bukan cuma urutan hasil compress.
    Perlu untuk custom union (HyDE) dan nanti relevance score extraction
    + threshold fallback DuckDuckGo (Advanced).

    Return: list (Document, score) terurut score descending.
    """
    if not documents:
        return []
    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.score(pairs)
    return sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)


def rerank_documents(
    reranker: HuggingFaceCrossEncoder,
    query: str,
    documents: list[Document],
    top_n: int = 3,
) -> list[Document]:
    """Convenience wrapper: rerank_with_scores() lalu ambil Top-N Document saja."""
    ranked = rerank_with_scores(reranker, query, documents)
    return [doc for doc, _ in ranked[:top_n]]
