"""
metadata.py

Utilitas untuk memperkaya Document dengan metadata `uu_number` dan
`pasal_refs`, serta memfilter dokumen berdasarkan nomor UU, pada
pipeline RAG (Legal AI Assistant).
"""

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
    """Menambahkan metadata `uu_number` dan `pasal_refs` ke tiap Document.

    Dipanggil SEBELUM splitting (parent/child), supaya metadata ini
    ter-carry otomatis ke semua turunan chunk lewat mekanisme
    metadata-copy bawaan LangChain splitter.

    `uu_number` di-mapping dari `source_file` (nama file), BUKAN
    di-derive dari regex isi teks -- nomor UU tidak selalu muncul
    konsisten di tiap halaman. Fungsi ini sengaja tidak fallback diam-
    diam ke nilai seperti `"Unknown"` apabila `source_file` tidak ada
    di mapping, dan langsung raise `ValueError` -- supaya kegagalan
    mapping (misal untuk dokumen PP 35/2021 yang dipakai test case
    wajib) langsung terlihat saat proses enrichment, bukan baru
    ketahuan belakangan saat sitasi hasil retrieval kosong atau salah.

    `pasal_refs` diekstrak lewat regex dari isi halaman ASLI (sebelum
    split), lalu disimpan sebagai string comma-separated -- bukan
    `list[str]` -- karena ChromaDB hanya menerima metadata value
    bertipe scalar (str/int/float/bool/None); menyimpan sebagai list
    akan menyebabkan `ValueError` saat ingest ke vectorstore.

    Trade-off yang perlu diketahui:
    - Granularitas `pasal_refs` adalah per halaman, bukan per chunk
      hasil split. Apabila satu halaman menyebut beberapa pasal,
      seluruh chunk turunan halaman tersebut akan mewarisi
      `pasal_refs` yang sama, walau chunk spesifiknya hanya membahas
      salah satu pasal.
    - Urutan nomor pasal dalam string hasil (`", ".join(sorted(...))`)
      mengikuti lexicographic (string) sort, bukan urutan numerik --
      misal `"10"` akan muncul sebelum `"2"`. Ini tidak memengaruhi
      korektnesnya data (himpunan nomor pasal yang tercatat tetap
      benar) maupun hasil retrieval/filtering (yang memakai
      `uu_number`, bukan `pasal_refs`), namun urutannya kurang natural
      apabila dibaca langsung sebagai teks sitasi.

    Args:
        documents: List `Document` hasil `load_pdfs`, sebelum splitting.

    Returns:
        List `Document` yang sama (dimodifikasi in-place), dengan
        metadata `uu_number` dan `pasal_refs` ditambahkan.

    Raises:
        ValueError: Apabila `source_file` suatu Document tidak
            ditemukan di `FILENAME_TO_UU_NUMBER`.
    """
    for doc in documents:
        source_file = doc.metadata.get("source_file", "")
        uu_number = FILENAME_TO_UU_NUMBER.get(source_file)
        if uu_number is None:
            raise ValueError(
                f"'{source_file}' tidak ada di FILENAME_TO_UU_NUMBER mapping. "
                f"Cek nama file persis di PDF_DIR atau update mapping di src/rag/metadata.py."
            )
        doc.metadata["uu_number"] = uu_number

        matches = PASAL_PATTERN.findall(doc.page_content)
        # Disimpan sebagai string comma-separated (bukan list) -- ChromaDB
        # hanya terima metadata value scalar (str/int/float/bool/None),
        # list akan raise ValueError saat ingest ke vectorstore.
        doc.metadata["pasal_refs"] = ", ".join(sorted(set(matches))) if matches else ""

    return documents


def filter_by_uu_number(documents: list[Document], uu_number: str) -> list[Document]:
    """Memfilter dokumen/chunk berdasarkan `uu_number` secara eksak.

    Berguna apabila query secara eksplisit menyebut UU tertentu
    (metadata filtering). Hanya berfungsi dengan benar setelah
    `enrich_documents` dipanggil pada `documents` yang sama.

    Args:
        documents: List `Document` yang sudah diperkaya lewat
            `enrich_documents`.
        uu_number: Nilai `uu_number` yang dicari, harus persis sama
            dengan salah satu value di `FILENAME_TO_UU_NUMBER`.

    Returns:
        List `Document` yang metadata `uu_number`-nya cocok persis
        dengan `uu_number` yang diberikan.
    """
    return [d for d in documents if d.metadata.get("uu_number") == uu_number]
