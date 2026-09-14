from IPython.display import Markdown, display
from langchain_core.documents import Document

from src.rag.chunking import build_splitters, log_chunking_config
from src.rag.generation import (
    SYSTEM_PROMPT_RAG,
    build_prompt_runnable,
    build_text_generation_pipeline,
    format_context,
    load_finetuned_model,
)
from src.rag.hyde import build_hyde_pipeline, generate_hypothetical_answers
from src.rag.ingestion import load_pdfs, validate_pdf_count
from src.rag.metadata import enrich_documents
from src.rag.retrievers import (
    build_bm25_retriever,
    build_dense_retriever,
    build_ensemble_retriever,
    build_reranked_retriever,
    build_reranker,
    dedup_documents,
    ingest_into_dense_retriever,
    print_retrieved_docs,
    rerank_documents,
    rerank_with_scores,
)
from src.rag.vectorstore import build_embedding_model, build_vectorstore
from src.rag.web_fallback import duckduckgo_fallback_documents

VALID_RETRIEVER_MODES = ("dense", "hybrid", "hybrid_rerank")


class RAGPipeline:
    """
    Bungkus retriever + llm + prompt jadi satu objek dengan interface
    sederhana: generate(query) -> {"answer": str, "sources": list[Document],
    "used_fallback": bool, "top_score": float | None}.

    Sengaja BUKAN pakai LCEL chain (`|`) murni -- karena kita butuh akses
    eksplisit ke `docs` hasil retrieval (untuk ditampilkan sebagai sitasi
    terpisah dari jawaban), dan LCEL chain murni bikin retriever di-invoke
    2x (sekali di dalam chain, sekali lagi manual buat nampilin sumber) --
    boros, dan berisiko dapat hasil retrieval yang beda kalau ada
    non-determinism. Di sini retrieval cuma dipanggil sekali per query.

    Empat mode retrieval, saling eksklusif lewat kombinasi use_hyde /
    use_fallback:
    - use_hyde=False, use_fallback=False -> retriever.invoke(query) polos
    - use_hyde=True,  use_fallback=False -> union HyDE, rerank sekali,
      TANPA threshold check
    - use_hyde=False, use_fallback=True  -> retrieve query asli, rerank,
      threshold check -> fallback DuckDuckGo kalau di bawah threshold
    - use_hyde=True,  use_fallback=True  -> union HyDE dulu, BARU rerank +
      threshold check di hasil union itu
    """

    def __init__(
        self,
        retriever,
        llm,
        tokenizer,
        system_prompt: str = SYSTEM_PROMPT_RAG,
        use_hyde: bool = False,
        hyde_llm=None,
        reranker=None,
        rerank_top_n: int = 3,
        hyde_n: int = 2,
        use_fallback: bool = False,
        fallback_threshold: float = 0.0,
    ):
        # Guard eksplisit -- fail fast dengan pesan jelas kalau dependency
        # yang dibutuhkan sebuah mode lupa di-pass, DAN kamu construct
        # RAGPipeline langsung (bukan lewat build_pipeline(), yang selalu
        # otomatis nyediain dependency ini). Tanpa guard ini, lupa pass
        # reranker/hyde_llm bakal crash AttributeError di tengah generate()
        # dengan pesan yang gak jelas asal-usulnya.
        if use_fallback and reranker is None:
            raise ValueError(
                "use_fallback=True butuh reranker (untuk ekstrak relevance "
                "score Top-1). Pass instance reranker, biasanya dari "
                "build_retrievers()['reranker']."
            )
        if use_hyde and hyde_llm is None:
            raise ValueError(
                "use_hyde=True butuh hyde_llm. Pass instance dari "
                "build_hyde_pipeline(llm)."
            )

        self.retriever = retriever
        self.llm = llm
        self.tokenizer = tokenizer
        self.prompt_runnable = build_prompt_runnable(tokenizer, system_prompt)
        self.use_hyde = use_hyde
        self.hyde_llm = hyde_llm
        self.reranker = reranker
        self.rerank_top_n = rerank_top_n
        self.hyde_n = hyde_n
        self.use_fallback = use_fallback
        self.fallback_threshold = fallback_threshold

    def generate(self, query: str) -> dict:
        if self.use_fallback:
            fallback_result = self._retrieve_with_optional_hyde_and_fallback(query)
            docs = fallback_result["docs"]
        else:
            docs = (
                self._retrieve_with_hyde(query)
                if self.use_hyde
                else self.retriever.invoke(query)
            )
            fallback_result = {"used_fallback": False, "top_score": None}

        context = format_context(docs)
        prompt = self.prompt_runnable.invoke({"context": context, "question": query})
        raw_output = self.llm.invoke(prompt)
        return {
            "answer": raw_output.strip(),
            "sources": docs,
            "used_fallback": fallback_result["used_fallback"],
            "top_score": fallback_result["top_score"],
        }

    def _retrieve_with_hyde(self, query: str) -> list[Document]:
        """
        self.retriever di sini HARUS base retriever pre-rerank (mis. "hybrid"),
        bukan "hybrid_rerank" -- karena rerank dilakukan manual SEKALI di
        bawah, setelah union semua kandidat (query asli + tiap hypothesis).
        Rerank per-hypothesis lalu di-union itu lebih mahal (banyak forward
        pass cross-encoder) dan hasilnya beberapa ranking terpisah yang
        digabung apa adanya -- bukan satu ranking konsisten.
        """
        hyde_answers = generate_hypothetical_answers(
            query, self.hyde_llm, self.tokenizer, n=self.hyde_n
        )
        doc_lists = [self.retriever.invoke(query)] + [
            self.retriever.invoke(ans) for ans in hyde_answers
        ]
        combined = dedup_documents(doc_lists)
        if self.reranker is not None:
            return rerank_documents(
                self.reranker, query, combined, top_n=self.rerank_top_n
            )
        return combined[: self.rerank_top_n]

    def _retrieve_with_optional_hyde_and_fallback(self, query: str) -> dict:
        """
        Kalau use_hyde=True: union HyDE dulu, BARU cek threshold di hasil
        rerank union itu. Kalau use_hyde=False: threshold dicek langsung
        di hasil retrieve query asli. reranker WAJIB ada kalau
        use_fallback=True -- sekarang di-enforce di __init__ lewat
        ValueError kalau lupa pasang (lihat guard di atas), bukan gagal
        diam-diam di sini.
        """
        if self.use_hyde:
            hyde_answers = generate_hypothetical_answers(
                query, self.hyde_llm, self.tokenizer, n=self.hyde_n
            )
            doc_lists = [self.retriever.invoke(query)] + [
                self.retriever.invoke(ans) for ans in hyde_answers
            ]
            candidates = dedup_documents(doc_lists)
        else:
            candidates = self.retriever.invoke(query)

        ranked = rerank_with_scores(self.reranker, query, candidates)
        top_score = ranked[0][1] if ranked else None

        if top_score is None or top_score < self.fallback_threshold:
            docs = duckduckgo_fallback_documents(query, max_results=self.rerank_top_n)
            return {"docs": docs, "used_fallback": True, "top_score": top_score}

        docs = [doc for doc, _ in ranked[: self.rerank_top_n]]
        return {"docs": docs, "used_fallback": False, "top_score": top_score}


def build_retrievers(
    pdf_dir: str,
    ensemble_weights: tuple[float, float] = (0.5, 0.5),
    reranker_top_n: int = 3,
) -> dict:
    """
    Bangun ketiga retriever_mode SEKALIGUS dari SATU proses ingestion
    (load PDF, chunking, embedding, ingest ke dense retriever) -- bukan
    diulang per mode seperti desain sebelumnya.

    FIX untuk OOM: build_pipeline() versi sebelumnya dipanggil 3x terpisah
    di notebook untuk bandingin retriever_mode, dan tiap panggilan itu
    nge-RELOAD GENERATOR MODEL dari HF Hub -- padahal generator sama
    sekali gak dibutuhkan untuk ablation retrieval (sanity_check_retrieval
    cuma invoke retriever, gak pernah invoke llm). Akibatnya 3 instance
    generator model (+ embedding model) numpuk di GPU memory tanpa pernah
    dibebaskan, sampai OOM pas load instance ke-3.

    Solusi: pisahkan proses build retriever (murah, gak butuh generator)
    dari load generator (mahal, sekali aja). Bonus: "dense", "hybrid", dan
    "hybrid_rerank" sebenarnya bertingkat (hybrid dibangun DI ATAS
    dense_retriever yang sama, hybrid_rerank DI ATAS hybrid yang sama) --
    jadi PDF+embedding cukup diproses sekali, dipakai bersama ketiganya,
    bukan re-embed dokumen yang sama 3x.

    Return: dict {"dense": ..., "hybrid": ..., "hybrid_rerank": ...,
    "reranker": ...}
    """
    documents = load_pdfs(pdf_dir)
    validate_pdf_count(documents, expected_files=4)
    documents = enrich_documents(documents)

    parent_splitter, child_splitter = build_splitters()
    log_chunking_config()

    embedding_model = build_embedding_model()
    vectorstore = build_vectorstore(embedding_model)

    dense_retriever = build_dense_retriever(
        vectorstore, parent_splitter, child_splitter
    )
    ingest_into_dense_retriever(dense_retriever, documents)

    bm25_retriever = build_bm25_retriever(documents, child_splitter)
    hybrid_retriever = build_ensemble_retriever(
        bm25_retriever, dense_retriever, ensemble_weights
    )

    reranker = build_reranker()
    hybrid_rerank_retriever = build_reranked_retriever(
        hybrid_retriever, reranker, top_n=reranker_top_n
    )

    return {
        "dense": dense_retriever,
        "hybrid": hybrid_retriever,
        "hybrid_rerank": hybrid_rerank_retriever,
        "reranker": reranker,
    }


def build_generator(hf_repo_id: str, hf_token: str | None = None):
    """
    Load generator SEKALI, dipakai ulang untuk retriever_mode manapun.
    Jangan panggil berkali-kali dalam satu sesi kernel kecuali memang mau
    ganti model (misal eksperimen run1 vs run2) -- ini komponen paling
    berat di GPU memory.
    """
    model, tokenizer = load_finetuned_model(hf_repo_id, hf_token=hf_token)
    llm = build_text_generation_pipeline(model, tokenizer)
    return llm, tokenizer


def build_pipeline(
    pdf_dir: str,
    hf_repo_id: str,
    retriever_mode: str = "hybrid_rerank",
    ensemble_weights: tuple[float, float] = (0.5, 0.5),
    reranker_top_n: int = 3,
    hf_token: str | None = None,
    use_hyde: bool = False,
    hyde_n: int = 2,
    use_fallback: bool = False,
    fallback_threshold: float = 0.0,
) -> RAGPipeline:
    """
    Convenience wrapper: bangun SATU retriever_mode + generator jadi
    RAGPipeline siap pakai. Cocok untuk pemakaian TUNGGAL (misal section
    "Full Pipeline untuk Interactive Use" di notebook, atau
    05_rag_final_evaluation.ipynb).

    Untuk BANDINGIN beberapa retriever_mode sekaligus (ablation study),
    JANGAN panggil fungsi ini berkali-kali -- pakai build_retrievers() +
    build_generator() terpisah, supaya PDF gak di-ingest ulang dan
    generator gak di-load ulang tiap mode (itu penyebab OOM sebelumnya).

    use_hyde=True: retriever_mode diabaikan -- HyDE selalu pakai base
    "hybrid" sebagai sumber kandidat (rerank dilakukan manual setelah
    union, bukan lewat mode "hybrid_rerank" yang reranking-nya sudah
    dibungkus di dalam retriever object).

    use_fallback=True: WAJIB dibarengi reranker (otomatis diambil dari
    build_retrievers()["reranker"] -- kamu gak perlu pass manual). Kalau
    use_fallback=True tapi use_hyde=False, retriever yang dipakai tetap
    base "hybrid" (bukan retriever_mode pilihan kamu) -- alasan sama
    dengan HyDE: threshold check butuh raw score dari rerank_with_scores(),
    bukan retriever yang reranking-nya sudah dibungkus di dalam object
    (ContextualCompressionRetriever tidak expose raw score keluar).

    fallback_threshold: skor minimum top-1 reranker (skala tergantung
    model, untuk bge-reranker-base umumnya logit belum di-sigmoid).
    Default 0.0 di sini BELUM dikalibrasi secara empiris terhadap
    distribusi skor query in-domain vs out-of-domain -- keputusan sadar
    karena keterbatasan waktu (dicatat sebagai known limitation di
    README), BUKAN klaim bahwa 0.0 adalah nilai optimal.
    """
    if retriever_mode not in VALID_RETRIEVER_MODES:
        raise ValueError(
            f"retriever_mode harus salah satu dari {VALID_RETRIEVER_MODES}, "
            f"dapat: {retriever_mode!r}"
        )

    retrievers = build_retrievers(pdf_dir, ensemble_weights, reranker_top_n)
    llm, tokenizer = build_generator(hf_repo_id, hf_token=hf_token)

    if use_hyde or use_fallback:
        # Baik HyDE maupun fallback butuh base retriever pre-rerank
        # ("hybrid") + akses raw score reranker -- lihat docstring di atas.
        hyde_llm = build_hyde_pipeline(llm) if use_hyde else None
        return RAGPipeline(
            retriever=retrievers["hybrid"],
            llm=llm,
            tokenizer=tokenizer,
            use_hyde=use_hyde,
            hyde_llm=hyde_llm,
            reranker=retrievers["reranker"],
            rerank_top_n=reranker_top_n,
            hyde_n=hyde_n,
            use_fallback=use_fallback,
            fallback_threshold=fallback_threshold,
        )

    retriever = retrievers[retriever_mode]
    return RAGPipeline(retriever=retriever, llm=llm, tokenizer=tokenizer)


def sanity_check_retrieval(retriever, query: str) -> None:
    """
    Verifikasi retriever jalan -- requirement eksplisit Basic ("uji
    retrieval pada query relevan, tampilkan hasil chunk").

    FIX: terima retriever LANGSUNG (bukan RAGPipeline seperti sebelumnya)
    -- verifikasi retrieval gak butuh generator sama sekali, jadi bisa
    dipanggil murah tanpa perlu build_pipeline() penuh (yang otomatis
    ikut load generator).
    """
    docs = retriever.invoke(query)
    print(f"Query: {query}\n")
    print_retrieved_docs(docs)


def _format_source_line(doc: Document, index: int) -> str:
    if doc.metadata.get("source_type") == "web":
        return (
            f"{index}. [Web] {doc.metadata.get('title', '?')} "
            f"({doc.metadata.get('url', '?')})"
        )

    line = (
        f"{index}. {doc.metadata.get('uu_number', '?')}, "
        f"{doc.metadata.get('source_file', '?')}"
    )
    if doc.metadata.get("page") is not None:
        line += f", halaman {doc.metadata['page']}"
    if doc.metadata.get("pasal_refs"):
        line += f" — Pasal: {doc.metadata['pasal_refs']}"
    return line


def interactive_loop(pipeline: RAGPipeline) -> None:
    """
    Interface wajib Basic: input() + IPython.display.Markdown.

    Ketik 'exit' atau 'quit' untuk keluar dari loop -- tanpa exit
    condition, ini technically infinite loop yang cuma bisa dihentikan
    dengan interrupt kernel, kurang enak untuk demo ke penilai.
    """
    print("Legal AI Assistant -- ketik 'exit' atau 'quit' untuk keluar.\n")
    while True:
        query = input("Pertanyaan: ").strip()
        if query.lower() in ("exit", "quit"):
            print("Selesai.")
            break
        if not query:
            continue

        result = pipeline.generate(query)

        display(Markdown(f"**Jawaban:**\n\n{result['answer']}"))

        if result.get("used_fallback"):
            display(
                Markdown(
                    f"*(Skor relevansi dokumen lokal di bawah threshold "
                    f"[{result.get('top_score')}] -- jawaban di atas pakai "
                    f"fallback DuckDuckGo, bukan dokumen UU lokal.)*"
                )
            )

        sources_md = "\n".join(
            _format_source_line(doc, i)
            for i, doc in enumerate(result["sources"], start=1)
        )
        display(Markdown(f"**Sumber Referensi:**\n\n{sources_md}"))
