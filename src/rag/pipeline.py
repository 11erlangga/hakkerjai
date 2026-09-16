"""
pipeline.py

Kelas `RAGPipeline` dan fungsi-fungsi pembangun (`build_retrievers`,
`build_generator`, `build_pipeline`) untuk merangkai retriever, generator
hasil fine-tuning sendiri, dan prompt runnable menjadi satu pipeline RAG
siap pakai (Legal AI Assistant), termasuk interface Interactive Python
Loop untuk demo.
"""

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
    """Membungkus retriever, llm, dan prompt menjadi satu objek pipeline.

    Interface sederhana: `generate(query) -> {"answer": str, "sources":
    list[Document], "used_fallback": bool, "top_score": float | None}`.

    Sengaja BUKAN memakai LCEL chain (`|`) murni -- karena dibutuhkan
    akses eksplisit ke `docs` hasil retrieval (untuk ditampilkan sebagai
    sitasi terpisah dari jawaban), dan LCEL chain murni akan membuat
    retriever di-invoke dua kali (sekali di dalam chain, sekali lagi
    manual untuk menampilkan sumber) -- boros, dan berisiko mendapat
    hasil retrieval yang berbeda apabila ada non-determinism. Di sini
    retrieval hanya dipanggil sekali per query.

    Empat mode retrieval, saling eksklusif lewat kombinasi `use_hyde` /
    `use_fallback`:
    - `use_hyde=False, use_fallback=False` -> `retriever.invoke(query)`
      polos.
    - `use_hyde=True, use_fallback=False` -> union HyDE, rerank sekali,
      TANPA threshold check.
    - `use_hyde=False, use_fallback=True` -> retrieve query asli, rerank,
      threshold check -> fallback DuckDuckGo apabila di bawah threshold.
    - `use_hyde=True, use_fallback=True` -> union HyDE dulu, BARU rerank
      dan threshold check pada hasil union tersebut.

    PENTING -- kontrak `retriever` yang di-pass ke constructor: apabila
    `use_hyde=True` dan/atau `use_fallback=True`, parameter `retriever`
    WAJIB berupa base retriever pre-rerank (misal hasil `build_retrievers
    ()["hybrid"]`), BUKAN retriever yang sudah membungkus reranking di
    dalamnya (misal `"hybrid_rerank"`, berupa
    `ContextualCompressionRetriever`). Reranking untuk kedua mode
    tersebut dilakukan secara manual sekali di `_retrieve_with_hyde` /
    `_retrieve_with_optional_hyde_and_fallback`, setelah union kandidat
    dari query asli dan hypothesis. Apabila retriever yang di-pass sudah
    membungkus reranking sendiri, hasilnya adalah double-rerank yang
    TIDAK memicu error apa pun, namun menghasilkan skor dan urutan
    dokumen yang keliru secara diam-diam. Constraint ini TIDAK di-enforce
    lewat kode (tidak ada pengecekan tipe retriever di `__init__`) --
    `build_pipeline()` selalu memenuhi kontrak ini secara otomatis,
    sehingga constraint ini hanya relevan apabila `RAGPipeline`
    di-construct secara manual di luar `build_pipeline()`.
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
        """Membangun instance `RAGPipeline`.

        Args:
            retriever: Retriever dasar. Lihat catatan kontrak retriever
                pada docstring kelas apabila `use_hyde` atau
                `use_fallback` bernilai True.
            llm: `HuggingFacePipeline` generator, hasil
                `build_text_generation_pipeline`.
            tokenizer: Tokenizer yang berpasangan dengan `llm`.
            system_prompt: System prompt untuk prompt runnable.
            use_hyde: Aktifkan query transformation HyDE.
            hyde_llm: Wajib diisi apabila `use_hyde=True`, instance dari
                `build_hyde_pipeline(llm)`.
            reranker: Wajib diisi apabila `use_fallback=True`, dipakai
                untuk mengekstrak relevance score Top-1.
            rerank_top_n: Jumlah dokumen yang diambil setelah rerank.
            hyde_n: Jumlah hypothesis yang dihasilkan HyDE.
            use_fallback: Aktifkan fallback ke DuckDuckGo apabila skor
                top-1 reranker di bawah `fallback_threshold`.
            fallback_threshold: Ambang skor top-1 reranker untuk memicu
                fallback.

        Raises:
            ValueError: Apabila `use_fallback=True` tapi `reranker` tidak
                diberikan, atau `use_hyde=True` tapi `hyde_llm` tidak
                diberikan. Guard ini fail-fast dengan pesan jelas,
                dipasang khusus untuk kasus `RAGPipeline` di-construct
                langsung (bukan lewat `build_pipeline()`, yang selalu
                otomatis menyediakan dependency ini) -- tanpa guard ini,
                dependency yang lupa di-pass akan menyebabkan
                `AttributeError` di tengah `generate()` dengan pesan yang
                tidak jelas asal-usulnya.
        """
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
        """Menjalankan satu siklus retrieval + generation untuk `query`.

        Args:
            query: Pertanyaan dari pengguna.

        Returns:
            Dict dengan key `"answer"` (str), `"sources"`
            (list[Document]), `"used_fallback"` (bool), dan `"top_score"`
            (float | None).
        """
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
        """Melakukan retrieval dengan union HyDE, lalu rerank sekali.

        `self.retriever` di sini HARUS berupa base retriever pre-rerank
        (misal `"hybrid"`), BUKAN `"hybrid_rerank"` -- lihat catatan
        kontrak retriever pada docstring kelas `RAGPipeline`. Rerank
        dilakukan manual SEKALI di bawah, setelah union semua kandidat
        (query asli + tiap hypothesis). Rerank per-hypothesis lalu
        di-union akan lebih mahal (banyak forward pass cross-encoder)
        dan hasilnya berupa beberapa ranking terpisah yang digabung apa
        adanya -- bukan satu ranking yang konsisten.

        Args:
            query: Pertanyaan dari pengguna.

        Returns:
            List `Document` hasil union HyDE, sudah di-rerank (apabila
            `self.reranker` tersedia) dan dipotong sampai
            `self.rerank_top_n`.
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
        """Melakukan retrieval dengan threshold check dan fallback web.

        Apabila `self.use_hyde=True`: union HyDE dilakukan dulu, BARU
        threshold dicek pada hasil rerank union tersebut. Apabila
        `self.use_hyde=False`: threshold dicek langsung pada hasil
        retrieve query asli. `self.reranker` WAJIB tersedia apabila
        `self.use_fallback=True` -- sudah di-enforce di `__init__` lewat
        `ValueError` apabila lupa dipasang (lihat guard pada
        `__init__`), sehingga tidak gagal diam-diam di sini.

        Args:
            query: Pertanyaan dari pengguna.

        Returns:
            Dict dengan key `"docs"` (list[Document]), `"used_fallback"`
            (bool), dan `"top_score"` (float | None).
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
    """Membangun ketiga retriever_mode sekaligus dari satu proses ingestion.

    Load PDF, chunking, embedding, dan ingest ke dense retriever hanya
    dilakukan SEKALI, bukan diulang per mode. Generator model TIDAK
    dimuat di fungsi ini -- pemisahan ini penting karena ablation
    retrieval (`sanity_check_retrieval`) hanya memanggil retriever, tidak
    pernah memanggil llm, sehingga generator sama sekali tidak dibutuhkan
    untuk membandingkan retriever_mode. Apabila generator ikut dimuat di
    sini dan fungsi ini dipanggil berkali-kali untuk tiap mode yang mau
    dibandingkan, generator akan ter-reload berulang tanpa pernah
    dibebaskan dari GPU memory, berisiko OOM.

    "dense", "hybrid", dan "hybrid_rerank" bertingkat satu sama lain
    (hybrid dibangun DI ATAS dense_retriever yang sama, hybrid_rerank DI
    ATAS hybrid yang sama) -- sehingga PDF dan embedding cukup diproses
    sekali dan dipakai bersama ketiganya, bukan di-re-embed untuk
    masing-masing mode.

    Args:
        pdf_dir: Path direktori PDF UU.
        ensemble_weights: Tuple bobot `(bm25_weight, dense_weight)` untuk
            ensemble retriever.
        reranker_top_n: Jumlah dokumen yang diambil setelah rerank pada
            `hybrid_rerank`.

    Returns:
        Dict dengan key `"dense"`, `"hybrid"`, `"hybrid_rerank"` (ketiganya
        retriever), dan `"reranker"` (instance reranker mentah, dipakai
        ulang untuk HyDE dan fallback threshold check).
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
    """Memuat generator satu kali, untuk dipakai ulang di retriever_mode manapun.

    Jangan dipanggil berkali-kali dalam satu sesi kernel kecuali memang
    ingin mengganti model (misal membandingkan eksperimen run1 vs run2)
    -- ini komponen paling berat penggunaan GPU memory di seluruh
    pipeline.

    Args:
        hf_repo_id: Repo HuggingFace Hub hasil fine-tuning sendiri.
        hf_token: Token HuggingFace, dibutuhkan apabila repo bersifat
            private.

    Returns:
        Tuple `(llm, tokenizer)` -- `llm` berupa `HuggingFacePipeline`
        siap pakai, `tokenizer` berpasangan dengannya.
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
    """Membangun satu retriever_mode + generator menjadi RAGPipeline siap pakai.

    Cocok untuk pemakaian tunggal (misal section "Full Pipeline untuk
    Interactive Use" pada notebook, atau notebook evaluasi akhir).

    Untuk membandingkan beberapa retriever_mode sekaligus (ablation
    study), JANGAN memanggil fungsi ini berkali-kali -- gunakan
    `build_retrievers()` dan `build_generator()` secara terpisah, supaya
    PDF tidak di-ingest ulang dan generator tidak di-load ulang untuk
    tiap mode.

    `use_hyde=True`: `retriever_mode` diabaikan -- HyDE selalu memakai
    base `"hybrid"` sebagai sumber kandidat (rerank dilakukan manual
    setelah union, bukan lewat mode `"hybrid_rerank"` yang reranking-nya
    sudah dibungkus di dalam retriever object).

    `use_fallback=True`: WAJIB dibarengi reranker, yang otomatis diambil
    dari `build_retrievers()["reranker"]` (tidak perlu di-pass manual).
    Apabila `use_fallback=True` tapi `use_hyde=False`, retriever yang
    dipakai tetap base `"hybrid"` (bukan `retriever_mode` pilihan) --
    alasan sama dengan HyDE: threshold check membutuhkan raw score dari
    `rerank_with_scores()`, bukan retriever yang reranking-nya sudah
    dibungkus di dalam object (`ContextualCompressionRetriever` tidak
    mengekspos raw score keluar).

    Args:
        pdf_dir: Path direktori PDF UU.
        hf_repo_id: Repo HuggingFace Hub hasil fine-tuning sendiri.
        retriever_mode: Salah satu dari `VALID_RETRIEVER_MODES`.
            Diabaikan apabila `use_hyde=True` atau `use_fallback=True`.
        ensemble_weights: Tuple bobot `(bm25_weight, dense_weight)`.
        reranker_top_n: Jumlah dokumen yang diambil setelah rerank.
        hf_token: Token HuggingFace, dibutuhkan apabila repo bersifat
            private.
        use_hyde: Aktifkan query transformation HyDE.
        hyde_n: Jumlah hypothesis yang dihasilkan HyDE.
        use_fallback: Aktifkan fallback DuckDuckGo berbasis threshold.
        fallback_threshold: Skor minimum top-1 reranker untuk lolos
            tanpa fallback (skala tergantung model -- untuk
            `bge-reranker-base` umumnya berupa logit mentah, belum
            di-sigmoid). Nilai default 0.0 di sini BELUM dikalibrasi
            secara empiris terhadap distribusi skor query in-domain vs
            out-of-domain -- keputusan sadar karena keterbatasan waktu,
            BUKAN klaim bahwa 0.0 adalah nilai optimal.

    Returns:
        Instance `RAGPipeline` siap dipakai untuk `generate(query)`.

    Raises:
        ValueError: Apabila `retriever_mode` bukan salah satu dari
            `VALID_RETRIEVER_MODES`.
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
    """Menguji retriever pada satu query dan mencetak hasilnya.

    Menerima retriever secara LANGSUNG (bukan `RAGPipeline`) -- verifikasi
    retrieval sama sekali tidak membutuhkan generator, sehingga dapat
    dipanggil secara murah tanpa perlu membangun `build_pipeline()` penuh
    (yang otomatis ikut memuat generator).

    Args:
        retriever: Retriever yang akan diuji (misal salah satu dari
            `build_retrievers()`).
        query: Query uji yang dianggap relevan dengan dokumen.

    Returns:
        None. Query dan hasil retrieval dicetak langsung ke stdout lewat
        `print_retrieved_docs`.
    """
    docs = retriever.invoke(query)
    print(f"Query: {query}\n")
    print_retrieved_docs(docs)


def _format_source_line(doc: Document, index: int) -> str:
    """Memformat satu Document menjadi satu baris teks sitasi sumber.

    Document hasil web fallback (`source_type="web"`) diformat sebagai
    judul + url. Document lokal diformat sebagai nomor UU + nama file,
    dengan tambahan nomor halaman (apabila tersedia di metadata) dan
    referensi pasal (apabila `pasal_refs` tidak kosong).

    Args:
        doc: Document sumber, hasil retrieval (lokal atau web fallback).
        index: Nomor urut tampilan (1-based), dipakai sebagai penomoran
            di awal baris.

    Returns:
        Satu baris string siap ditampilkan sebagai daftar sumber.
    """
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
    """Menjalankan loop tanya-jawab interaktif berbasis `input()`.

    Ketik `'exit'` atau `'quit'` untuk keluar dari loop -- tanpa exit
    condition, ini secara teknis merupakan infinite loop yang hanya bisa
    dihentikan dengan interrupt kernel.

    Args:
        pipeline: Instance `RAGPipeline` yang sudah siap dipakai.

    Returns:
        None. Jawaban dan sumber ditampilkan langsung lewat
        `IPython.display.Markdown`.
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
