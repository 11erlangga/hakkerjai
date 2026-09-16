"""
ingestion.py

Utilitas untuk memuat dokumen PDF UU dan memvalidasi kelengkapan file
sebelum masuk ke tahap chunking pada pipeline RAG (Legal AI Assistant).
"""

from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


def load_pdfs(pdf_dir: str, min_content_length: int = 20) -> list[Document]:
    """Memuat semua PDF di `pdf_dir` dan menandai sumbernya.

    Setiap Document diberi metadata `source_file` (nama file, sebagai
    identitas UU/PP terkait). Pencarian file bersifat case-insensitive
    untuk ekstensi `.pdf`/`.PDF`, dan urutan file di-sort agar log/debug
    konsisten antar run.

    Halaman dengan konten nyaris kosong (indikasi hasil scan/gambar
    tanpa OCR) diberi peringatan -- bukan auto-fix -- supaya diketahui
    sejak awal apabila ada bagian dokumen yang akan efektif invisible
    untuk retrieval nantinya.

    Args:
        pdf_dir: Path direktori yang berisi file-file PDF UU.
        min_content_length: Ambang jumlah karakter (setelah strip) yang
            dipakai untuk menandai suatu halaman sebagai "nyaris kosong".

    Returns:
        List gabungan seluruh `Document` dari semua file PDF di
        `pdf_dir`.
    """
    pdf_paths = sorted(
        {*Path(pdf_dir).glob("*.pdf"), *Path(pdf_dir).glob("*.PDF")},
        key=lambda p: p.name,
    )

    documents: list[Document] = []
    thin_content_count = 0

    for pdf_path in pdf_paths:
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()

        for doc in docs:
            # Normalize source path jadi nama file bersih.
            doc.metadata["source_file"] = pdf_path.name

            if len(doc.page_content.strip()) < min_content_length:
                thin_content_count += 1
                print(
                    f"[WARNING] Konten nyaris kosong (<{min_content_length} char) "
                    f"di {pdf_path.name}, halaman {doc.metadata.get('page', '?')}. "
                    f"Kemungkinan hasil scan/gambar tanpa OCR -- cek manual, "
                    f"karena halaman ini efektif invisible untuk retrieval."
                )

        documents.extend(docs)

    if thin_content_count > 0:
        print(
            f"\n[SUMMARY] Total {thin_content_count} halaman dengan konten "
            f"nyaris kosong dari {len(pdf_paths)} file PDF. "
            f"Pertimbangkan OCR manual kalau jumlah ini signifikan."
        )

    return documents


def validate_pdf_count(documents: list[Document], expected_files: int = 4) -> None:
    """Memastikan seluruh file PDF yang diharapkan berhasil termuat.

    Sanity check wajib sebelum lanjut ke tahap berikutnya: memastikan
    seluruh file (bukan sebagian, misal karena typo path atau file yang
    kebetulan tidak ter-mount) benar-benar terbaca. Jumlah `source_file`
    unik pada metadata dibandingkan dengan `expected_files`.

    Args:
        documents: List `Document` hasil `load_pdfs`.
        expected_files: Jumlah file PDF unik yang diharapkan terbaca.

    Raises:
        AssertionError: Apabila jumlah `source_file` unik pada
            `documents` tidak sama dengan `expected_files` -- dipakai
            supaya proses tidak diam-diam lanjut dengan data yang tidak
            lengkap.
    """
    unique_files = {doc.metadata.get("source_file") for doc in documents}
    assert len(unique_files) == expected_files, (
        f"Expected {expected_files} unique PDF files, but found {len(unique_files)}. "
        f"Unique files: {unique_files}"
    )
