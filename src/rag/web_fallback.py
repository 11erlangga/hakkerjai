"""
web_fallback.py

Utilitas fallback pencarian web (DuckDuckGo) untuk pipeline RAG (Legal
AI Assistant), dipakai ketika hasil retrieval dokumen lokal dinilai
tidak cukup relevan berdasarkan skor reranker.
"""

from langchain_core.documents import Document

from src.rag.retrievers import rerank_with_scores


def _duckduckgo_search(query: str, max_results: int = 3) -> list[dict]:
    """Wrapper tipis di atas `ddgs.DDGS().text()`.

    Package ini dulu bernama `duckduckgo-search`, sudah di-rename resmi
    menjadi `ddgs` (`pip install ddgs`, `from ddgs import DDGS`). Import
    dilakukan LOKAL di dalam fungsi (bukan top-level module) supaya
    modul ini tetap bisa di-import untuk keperluan lain (misal testing
    logic threshold memakai mock) tanpa langsung membutuhkan package
    `ddgs` ter-install.

    Args:
        query: Query pencarian.
        max_results: Jumlah hasil maksimum yang diambil.

    Returns:
        List dict hasil pencarian mentah dari `ddgs`.
    """
    from ddgs import DDGS

    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
    return list(results)


def duckduckgo_fallback_documents(query: str, max_results: int = 3) -> list[Document]:
    """Mengubah hasil pencarian DuckDuckGo menjadi pseudo-Document.

    Metadata hasil fungsi ini SENGAJA berbeda struktur dari dokumen
    lokal (`source_type="web"` beserta `title`/`url`, dibanding
    `uu_number`/`pasal_refs` pada dokumen lokal) -- konsumen hilir
    (`format_context`, `interactive_loop`, `_format_source_line`) harus
    melakukan branching berdasarkan `source_type` ini, bukan berasumsi
    semua Document memiliki struktur metadata yang sama.

    Field hasil `ddgs.text()` yang dipakai: `"title"`, `"href"`,
    `"body"` -- diakses memakai `.get()` secara defensif karena field
    ini tidak dijamin API-stable dalam jangka panjang (riwayat rename
    package dari `duckduckgo-search` menjadi `ddgs` adalah contoh nyata
    mengapa akses defensif diperlukan).

    Args:
        query: Query pencarian.
        max_results: Jumlah hasil maksimum yang diambil.

    Returns:
        List `Document` dengan `page_content` dari `body` (atau `title`
        sebagai fallback apabila `body` kosong), dan metadata
        `source_type="web"`, `title`, `url`.
    """
    results = _duckduckgo_search(query, max_results=max_results)
    docs = []
    for r in results:
        content = r.get("body", "") or r.get("title", "")
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source_type": "web",
                    "title": r.get("title", "(tanpa judul)"),
                    "url": r.get("href", r.get("url", "")),
                },
            )
        )
    return docs


def retrieve_with_fallback(
    query: str,
    retriever,
    reranker,
    threshold: float,
    top_n: int = 3,
) -> dict:
    """Melakukan retrieval dengan threshold check dan fallback DuckDuckGo.

    Mengekstrak relevance score dari Top-1 hasil reranker, lalu
    menerapkan aturan if-else: apabila skor di bawah `threshold`,
    dokumen lokal diabaikan dan hasil di-fallback ke pencarian
    DuckDuckGo.

    Duplikasi yang disengaja: fungsi ini merupakan versi demo standalone
    dari logic yang juga diimplementasikan di
    `RAGPipeline._retrieve_with_optional_hyde_and_fallback` (pada mode
    `use_hyde=False`) di `pipeline.py`. Fungsi ini dipertahankan
    terpisah -- dipanggil langsung di notebook (lihat NB04 Section 4)
    -- supaya logic if-else threshold dapat didemonstrasikan secara
    terisolasi tanpa perlu membangun `RAGPipeline` penuh. Konsekuensi:
    apabila logic threshold perlu diubah, perubahan harus diterapkan di
    KEDUA tempat, karena tidak ada pemanggilan silang antara keduanya.

    Kontrak retriever: `retriever` diasumsikan berupa base retriever
    PRE-RERANK (misal mode `"hybrid"`), bukan `"hybrid_rerank"` yang
    sudah membungkus reranking di dalam retriever object -- konsisten
    dengan pola yang sama pada HyDE (`_retrieve_with_hyde` di
    `pipeline.py`): base retriever mengambil kandidat mentah, rerank
    dilakukan manual sekali di titik keputusan, supaya skor mentahnya
    dapat diakses untuk threshold check, bukan hanya urutan hasil
    compress dari `ContextualCompressionRetriever`. Constraint ini tidak
    di-enforce lewat kode.

    Dikembalikan sebagai dict eksplisit (bukan hanya list Document)
    supaya pemanggil (notebook cell atau kode lain) dapat mencetak atau
    mencatat `top_score` dan `used_fallback` secara terpisah dari
    `docs` itu sendiri -- penting agar logic if-else terlihat jelas saat
    didemonstrasikan.

    Args:
        query: Query dari pengguna.
        retriever: Base retriever pre-rerank (lihat catatan kontrak di
            atas).
        reranker: Instance cross-encoder hasil `build_reranker`.
        threshold: Ambang skor top-1 reranker untuk memicu fallback.
        top_n: Jumlah dokumen yang dikembalikan.

    Returns:
        Dict dengan key `"docs"` (list[Document]), `"used_fallback"`
        (bool), dan `"top_score"` (float | None).
    """
    candidates = retriever.invoke(query)
    ranked = rerank_with_scores(reranker, query, candidates)
    top_score = ranked[0][1] if ranked else None

    if top_score is None or top_score < threshold:
        docs = duckduckgo_fallback_documents(query, max_results=top_n)
        return {"docs": docs, "used_fallback": True, "top_score": top_score}

    docs = [doc for doc, _ in ranked[:top_n]]
    return {"docs": docs, "used_fallback": False, "top_score": top_score}
