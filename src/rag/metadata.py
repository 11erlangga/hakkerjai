import re

from langchain_core.documents import Document

PASAL_PATTERN = re.compile(r"Pasal\s+(\d+[A-Za-z]?)", re.IGNORECASE)

FILENAME_TO_UU_NUMBER = {
    "PP Nomor 35 Tahun 2021.pdf": "PP No. 35 Tahun 2021",
    "PP Nomor 5 Tahun 2021.pdf": "PP No. 5 Tahun 2021",
    "PP Nomor 51 Tahun 2023.pdf": "PP No. 51 Tahun 2023",
    "UU Nomor 6 Tahun 2023.pdf": "UU No. 6 Tahun 2023",
}


def enrich_documents(documents: list[Document]) -> list[Document]:
    """
    Tambahkan metadata 'uu_number' dan 'pasal_refs' ke tiap Document,
    SEBELUM splitting (parent/child) -- supaya ke-carry otomatis ke semua
    turunan chunk lewat mekanisme metadata-copy bawaan LangChain splitter.

    'uu_number': di-mapping dari source_file (nama file), BUKAN di-derive
    dari regex isi teks -- nomor UU gak selalu muncul konsisten di tiap
    halaman. Raise ValueError kalau source_file gak ada di mapping --
    sengaja gak fallback diam-diam ke "Unknown", karena kalau file test
    case wajib (PP 35/2021) somehow gak ke-mapping, kamu harus tahu SEKARANG
    bukan pas debugging kenapa sitasi test case kosong/salah nanti.

    'pasal_refs': regex-extracted dari isi halaman ASLI (sebelum split).
    Trade-off yang perlu didokumentasikan di README: granularity-nya per
    halaman, bukan per chunk hasil split -- kalau satu halaman menyebut
    beberapa pasal, semua chunk turunan halaman itu bakal mewarisi
    pasal_refs yang sama walau chunk spesifiknya cuma bahas satu pasal.
    """
    for doc in documents:
        source_file = doc.metadata.get("source_file", "")
        uu_number = FILENAME_TO_UU_NUMBER.get(source_file)
        if uu_number is None:
            raise ValueError(
                f"'{source_file}' tidak ada di FILENAME_TO_UU_NUMBER mapping. "
                f"Cek nama file persis di PDF_DIR (sorted(Path(PDF_DIR).glob('*.pdf'))) "
                f"atau update mapping di src/rag/metadata.py."
            )
        doc.metadata["uu_number"] = uu_number

        matches = PASAL_PATTERN.findall(doc.page_content)
        doc.metadata["pasal_refs"] = sorted(set(matches)) if matches else []

    return documents


def filter_by_uu_number(documents: list[Document], uu_number: str) -> list[Document]:
    """
    Filter dokumen/chunk berdasarkan uu_number eksak -- berguna kalau
    query eksplisit sebut UU tertentu (metadata filtering, requirement
    Skilled). Hanya berfungsi setelah enrich_documents() dipanggil.
    """
    return [d for d in documents if d.metadata.get("uu_number") == uu_number]
