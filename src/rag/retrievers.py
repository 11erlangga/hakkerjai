"""
retrievers.py

Utilitas untuk membangun berbagai retriever (dense parent-child, BM25,
ensemble, reranked) dan operasi pendukungnya (dedup, reranking manual
dengan akses raw score) pada pipeline RAG (Legal AI Assistant).
"""

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
    """Membangun dense retriever dengan pola parent-child chunking.

    Child chunks di-embed dan di-search, namun yang dikembalikan ke LLM
    adalah parent chunk (konteks lebih utuh).

    `docstore` selalu in-memory (bukan `persist_directory` seperti
    Chroma) -- ini konsekuensi dari desain `ParentDocumentRetriever`:
    parent chunk disimpan sebagai objek Python biasa (bukan vector),
    sehingga apabila kernel Kaggle restart, docstore ini HILANG meskipun
    vectorstore-nya persist. Artinya `retriever.add_documents(...)`
    harus di-run ulang tiap sesi baru, walau Chroma collection-nya
    sendiri sudah ter-cache di disk. Ini bukan bug, melainkan trade-off
    yang perlu disadari sebelum men-debug "kenapa konteks kosong padahal
    vectorstore-nya sudah ter-load".

    Args:
        vectorstore: Vector store (Chroma) untuk menyimpan embedding
            child chunks.
        parent_splitter: Text splitter untuk chunk parent.
        child_splitter: Text splitter untuk chunk child.
        search_k: Jumlah kandidat child chunk yang diambil per pencarian.

    Returns:
        Instance `ParentDocumentRetriever` siap diisi lewat
        `ingest_into_dense_retriever`.
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
    """Mengisi dense retriever dengan dokumen, dalam batch bertahap.

    Memicu parent+child splitting internal `ParentDocumentRetriever`,
    mengisi vectorstore (child) dan docstore (parent) sekaligus. Dipanggil
    SEKALI per sesi kernel setelah retriever dibuat.

    Dokumen di-push per batch (bukan semua sekaligus) karena Chroma
    membatasi jumlah embedding yang bisa di-upsert dalam satu panggilan
    API (pada versi yang dipakai: maksimum 5461). Apabila seluruh parent
    documents (ratusan halaman dari 4 PDF UU) di-push sekaligus, jumlah
    child chunks yang dihasilkan bisa jauh melebihi limit tersebut --
    pernah terjadi: 6583 chunk vs limit 5461.

    `batch_size=50` di sini adalah jumlah PARENT-level input documents
    (halaman PDF) per batch, BUKAN jumlah child chunks -- karena jumlah
    child chunks yang dihasilkan per halaman tidak dikontrol langsung
    (tergantung isi halaman, bisa 1 chunk bisa belasan). 50 halaman per
    batch merupakan margin konservatif di bawah limit 5461, cukup aman
    kecuali halaman-halaman tersebut jauh lebih padat teks dari biasanya
    -- apabila masih terkena limit error, turunkan `batch_size` lebih
    kecil (misal 20).

    Args:
        retriever: Instance `ParentDocumentRetriever` hasil
            `build_dense_retriever`.
        documents: List `Document` (level halaman) yang akan di-ingest.
        batch_size: Jumlah dokumen level-parent per batch upsert.

    Returns:
        None. Progress tiap batch dicetak ke stdout.
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
    """Membangun BM25 retriever dari chunk berukuran child.

    Index BM25 dibangun dari chunk berukuran child (bukan raw page-level
    documents), supaya granularitas konsisten dengan dense retriever --
    saat di-ensemble nanti, kedua retriever "berbicara dalam unit yang
    sama" (chunk ~400 karakter), bukan BM25 mengembalikan satu halaman
    penuh sementara dense mengembalikan potongan kecil.

    Catatan: chunk di sini di-split langsung dari `documents` memakai
    `child_splitter`, TERPISAH dari child chunks yang dibuat secara
    internal oleh `ParentDocumentRetriever` (yang di-split dari parent
    chunks dulu, baru child). Batas chunk-nya karena itu tidak dijamin
    identik character-per-character dengan yang ada di dense index --
    namun ukuran dan overlap-nya sama, yang menjadi concern utama.

    Args:
        documents: List `Document` (level halaman) sumber.
        child_splitter: Text splitter berukuran child, sama dengan yang
            dipakai `build_dense_retriever`.
        k: Jumlah kandidat yang dikembalikan per pencarian.

    Returns:
        Instance `BM25Retriever` siap pakai.
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
    """Menggabungkan BM25 dan dense retriever menjadi satu ensemble.

    BM25 (keyword) kuat untuk istilah pasal/nomor UU yang eksak, dense
    retriever (semantic) kuat untuk pertanyaan berbahasa natural yang
    tidak persis memakai istilah dokumen.

    `weights=(0.5, 0.5)` merupakan titik awal netral, belum di-tuning
    secara kuantitatif -- bobot ditentukan secara eksplisit sebagai
    keputusan desain, bukan hasil pencarian nilai optimal. Apabila hasil
    retrieval kualitatif terlihat bias ke salah satu retriever, ini
    parameter pertama yang disarankan diubah.

    Args:
        bm25_retriever: Retriever BM25 hasil `build_bm25_retriever`.
        dense_retriever: Retriever dense hasil `build_dense_retriever`
            yang sudah di-ingest.
        weights: Tuple `(bm25_weight, dense_weight)`.

    Returns:
        Instance `EnsembleRetriever` yang menggabungkan keduanya.
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
    """Membungkus base retriever dengan cross-encoder reranker.

    `base_retriever` (biasanya ensemble) mengambil kandidat lebih banyak
    dulu (k=10 per retriever), lalu reranker membaca ulang tiap kandidat
    BERSAMA query-nya (cross-attention) -- lebih akurat namun lebih
    lambat dibanding scoring bi-encoder biasa.

    Args:
        base_retriever: Retriever dasar (biasanya hasil
            `build_ensemble_retriever`) yang kandidatnya akan di-rerank.
        reranker: Instance cross-encoder hasil `build_reranker`.
        top_n: Jumlah dokumen teratas yang dikembalikan setelah rerank.

    Returns:
        Instance `ContextualCompressionRetriever` yang membungkus
        reranking di dalam pemanggilan `.invoke()`-nya.
    """
    compressor = CrossEncoderReranker(model=reranker, top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def print_retrieved_docs(docs: list[Document]) -> None:
    """Mencetak hasil retrieval untuk inspeksi manual (sanity check).

    Untuk tiap dokumen, dicetak nomor urut, nama file sumber, nomor
    halaman, nomor UU/PP, referensi pasal yang terdeteksi (apabila ada),
    dan isi chunk itu sendiri -- membantu verifikasi bahwa retriever
    mengembalikan chunk yang relevan dan metadata-nya terisi benar.

    Args:
        docs: List `Document` hasil retrieval yang akan diinspeksi.

    Returns:
        None. Output dicetak langsung ke stdout.
    """
    for i, doc in enumerate(docs, start=1):
        print(f"--- Dokumen {i} ---")
        print(
            f"Sumber: {doc.metadata.get('source_file', '?')}, "
            f"halaman: {doc.metadata.get('page', '?')}"
        )
        print(
            f"UU/PP: {doc.metadata.get('uu_number', '?')}, "
            f"Pasal terdeteksi: {doc.metadata.get('pasal_refs') or '(tidak ada)'}"
        )
        print(doc.page_content)
        print()


def build_reranker(
    reranker_model_name: str = RERANKER_MODEL_NAME,
) -> HuggingFaceCrossEncoder:
    """Memuat cross-encoder reranker satu kali per sesi kernel.

    Instance ini di-reuse di `build_reranked_retriever` dan
    `rerank_with_scores` -- jangan dipanggil ulang, supaya tidak
    menumpuk dua instance model reranker di GPU memory tanpa perlu.

    Args:
        reranker_model_name: Nama model cross-encoder di HuggingFace Hub.

    Returns:
        Instance `HuggingFaceCrossEncoder` siap dipakai berulang.
    """
    return HuggingFaceCrossEncoder(model_name=reranker_model_name)


def dedup_documents(doc_lists: list[list[Document]]) -> list[Document]:
    """Menggabungkan beberapa list hasil retrieval dan menghapus duplikat.

    Dedup dilakukan berdasarkan 100 karakter pertama `page_content`.
    Urutan diprioritaskan sesuai urutan `doc_lists` yang di-pass (list
    pertama menang apabila ada duplikat).

    Known limitation: key dedup berbasis 100 karakter pertama adalah
    heuristik, bukan pembanding isi penuh. Dua chunk yang isinya
    berbeda namun kebetulan memiliki 100 karakter awal yang identik
    (misal pembukaan kalimat pasal yang berpola serupa, umum terjadi
    pada teks legal) akan dianggap duplikat, dan salah satunya akan
    di-drop secara diam-diam dari hasil union -- tanpa error maupun
    peringatan. Heuristik ini dipilih karena cukup efektif untuk kasus
    utama pemakaiannya (dokumen yang sama muncul berulang pada hasil
    retrieval query asli dan beberapa hypothesis HyDE), namun bukan
    jaminan dedup yang presisi berbasis isi penuh.

    Args:
        doc_lists: List berisi beberapa list `Document` yang akan
            digabung dan di-dedup.

    Returns:
        List `Document` gabungan tanpa duplikat, urutan mengikuti
        prioritas `doc_lists`.
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
    """Memberi skor tiap dokumen terhadap query, mengembalikan raw score.

    Scoring dilakukan secara manual lewat akses `.score()` langsung ke
    reranker (BUKAN lewat `ContextualCompressionRetriever`), supaya raw
    score-nya dapat diakses, bukan hanya urutan hasil compress. Ini
    dibutuhkan untuk custom union (HyDE) dan untuk ekstraksi relevance
    score beserta threshold fallback DuckDuckGo.

    Args:
        reranker: Instance cross-encoder hasil `build_reranker`.
        query: Query yang dipakai untuk scoring.
        documents: List `Document` kandidat yang akan diberi skor.

    Returns:
        List tuple `(Document, score)`, terurut skor menurun (descending).
        List kosong apabila `documents` kosong.
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
    """Melakukan rerank lalu mengambil Top-N Document saja.

    Convenience wrapper di atas `rerank_with_scores`, dipakai ketika
    raw score tidak dibutuhkan oleh pemanggil.

    Args:
        reranker: Instance cross-encoder hasil `build_reranker`.
        query: Query yang dipakai untuk scoring.
        documents: List `Document` kandidat yang akan di-rerank.
        top_n: Jumlah dokumen teratas yang dikembalikan.

    Returns:
        List `Document` sepanjang maksimum `top_n`, terurut relevansi
        menurun.
    """
    ranked = rerank_with_scores(reranker, query, documents)
    return [doc for doc, _ in ranked[:top_n]]
