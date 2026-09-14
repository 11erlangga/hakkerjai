from langchain_core.documents import Document

from src.rag.retrievers import rerank_with_scores


def _duckduckgo_search(query: str, max_results: int = 3) -> list[dict]:
    """
    Wrapper tipis di atas ddgs.DDGS().text() -- package ini dulu bernama
    duckduckgo-search, sudah di-rename resmi jadi `ddgs` (pip install ddgs,
    from ddgs import DDGS). Import di-lakukan LOKAL di dalam fungsi (bukan
    top-level module) supaya modul ini tetap bisa di-import untuk
    keperluan lain (mis. testing threshold logic pakai mock) tanpa
    langsung butuh package ddgs ter-install.
    """
    from ddgs import DDGS

    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
    return list(results)


def duckduckgo_fallback_documents(query: str, max_results: int = 3) -> list[Document]:
    """
    Wrap hasil DuckDuckGo jadi pseudo-Document, metadata SENGAJA beda
    struktur dari dokumen lokal (source_type="web" vs uu_number/pasal_refs)
    -- konsumen hilir (format_context, interactive_loop) harus branch
    berdasar source_type ini, bukan asumsi semua Document punya uu_number.

    Field hasil ddgs.text(): "title", "href", "body" -- pakai .get() defensif
    karena field ini gak dijamin API-stable jangka panjang (riwayat rename
    package ini contoh nyata kenapa perlu defensif).
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
    """
    Requirement Advanced brief: ekstrak relevance score dari Top-1 hasil
    reranker -> if-else -> kalau di bawah threshold, abaikan dokumen
    lokal, fallback ke DuckDuckGo Search.

    KEPUTUSAN DESAIN (perlu kamu setujui/debat, bukan otomatis benar):
    `retriever` di sini diasumsikan base retriever PRE-RERANK (mis. mode
    "hybrid") -- bukan "hybrid_rerank" yang sudah dibungkus reranking di
    dalam retriever object. Ini konsisten sama pola HyDE
    (_retrieve_with_hyde di pipeline.py): base retriever ambil kandidat
    mentah, rerank dilakukan MANUAL sekali di titik keputusan -- supaya
    skor mentahnya bisa diakses buat threshold check, bukan cuma urutan
    hasil compress dari ContextualCompressionRetriever.

    Return dict eksplisit (bukan cuma list Document) -- requirement
    Advanced perlu "kelihatan" logic if-else-nya pas didemo di notebook,
    jadi caller (notebook cell / RAGPipeline) bisa print/log top_score dan
    used_fallback secara terpisah dari docs itu sendiri.
    """
    candidates = retriever.invoke(query)
    ranked = rerank_with_scores(reranker, query, candidates)
    top_score = ranked[0][1] if ranked else None

    if top_score is None or top_score < threshold:
        docs = duckduckgo_fallback_documents(query, max_results=top_n)
        return {"docs": docs, "used_fallback": True, "top_score": top_score}

    docs = [doc for doc, _ in ranked[:top_n]]
    return {"docs": docs, "used_fallback": False, "top_score": top_score}
