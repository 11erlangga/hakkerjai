"""
chunking.py

Utilitas untuk membangun text splitter parent-child yang dipakai pada
pipeline chunking dokumen PDF UU (RAG - Legal AI Assistant).
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50


def build_splitters(
    parent_chunk_size: int = PARENT_CHUNK_SIZE,
    parent_overlap: int = PARENT_CHUNK_OVERLAP,
    child_chunk_size: int = CHILD_CHUNK_SIZE,
    child_overlap: int = CHILD_CHUNK_OVERLAP,
) -> tuple[RecursiveCharacterTextSplitter, RecursiveCharacterTextSplitter]:
    """Membangun pasangan text splitter untuk pola parent-child chunking.

    Nilai ukuran dan overlap chunk di-parameterize (bukan hardcode di
    tempat lain) agar mudah dieksperimen tanpa mengubah banyak file.
    Ukuran dan overlap harus eksplisit (bukan default tersembunyi) --
    lihat `log_chunking_config` untuk logging konfigurasi yang dipakai.

    Args:
        parent_chunk_size: Ukuran maksimum (karakter) untuk chunk
            parent -- potongan besar/halaman utuh yang dipakai sebagai
            context untuk LLM.
        parent_overlap: Overlap (karakter) antar chunk parent.
        child_chunk_size: Ukuran maksimum (karakter) untuk chunk child
            -- potongan kecil yang dipakai untuk vector search. Harus
            lebih kecil dari `parent_chunk_size`.
        child_overlap: Overlap (karakter) antar chunk child.

    Returns:
        Tuple `(parent_splitter, child_splitter)`, keduanya instance
        `RecursiveCharacterTextSplitter`.

    Raises:
        ValueError: Apabila `child_chunk_size >= parent_chunk_size`,
            karena chunk child yang lebih besar atau sama dengan parent
            tidak sesuai dengan pola parent-child chunking (child
            seharusnya potongan kecil di dalam parent).
    """
    if child_chunk_size >= parent_chunk_size:
        raise ValueError(
            f"child_chunk_size ({child_chunk_size}) harus lebih kecil "
            f"dari parent_chunk_size ({parent_chunk_size})"
        )

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return parent_splitter, child_splitter


def log_chunking_config(
    parent_chunk_size: int = PARENT_CHUNK_SIZE,
    parent_overlap: int = PARENT_CHUNK_OVERLAP,
    child_chunk_size: int = CHILD_CHUNK_SIZE,
    child_overlap: int = CHILD_CHUNK_OVERLAP,
) -> None:
    """Mencetak ukuran dan overlap chunk yang dipakai secara eksplisit.

    Konfigurasi chunking di-log eksplisit agar nilai yang benar-benar
    dipakai pada suatu run selalu terlihat jelas di output notebook,
    bukan tersembunyi di default argumen.

    Catatan desain: fungsi ini menerima angka konfigurasi secara
    langsung (bukan menerima objek splitter dan membaca atribut
    seperti `splitter.chunk_size`), karena `chunk_size` bukan
    merupakan atribut publik yang dijamin ada di semua versi
    LangChain (implementasi internal menyimpannya sebagai
    `_chunk_size`, atribut privat yang bisa berubah antar versi).
    Nilai yang di-log di sini harus sama persis dengan yang di-pass ke
    `build_splitters`.

    Args:
        parent_chunk_size: Ukuran chunk parent yang sedang dipakai.
        parent_overlap: Overlap chunk parent yang sedang dipakai.
        child_chunk_size: Ukuran chunk child yang sedang dipakai.
        child_overlap: Overlap chunk child yang sedang dipakai.

    Returns:
        None. Konfigurasi dicetak langsung ke stdout.
    """
    print(f"Parent chunk size: {parent_chunk_size}, overlap: {parent_overlap}")
    print(f"Child chunk size: {child_chunk_size}, overlap: {child_overlap}")
