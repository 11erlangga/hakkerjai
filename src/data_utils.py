"""
data_utils.py

Utilitas untuk memuat dan memformat dataset Alpaca-GPT4-Indonesian ke
format Chat Template (melalui tokenizer Unsloth). Modul ini dipakai
bersama di seluruh notebook eksperimen SFT agar logic mapping tetap
konsisten dan tidak terduplikasi antar eksperimen.
"""

import random

from datasets import Dataset, load_dataset

DATASET_NAME = "Ichsan2895/alpaca-gpt4-indonesian"

THINK_TEMPLATES = [
    "Pertanyaan ini meminta saya untuk {task_hint}. Saya akan menjawab secara langsung dan jelas.",
    "Untuk menjawab ini, saya perlu memahami inti permintaan terkait {task_hint}. Berikut jawabannya.",
    "Saya akan menyusun jawaban berdasarkan konteks yang diberikan terkait {task_hint}.",
    "Permintaan ini berkaitan dengan {task_hint}. Saya akan memberikan jawaban yang relevan dan ringkas.",
    "Berdasarkan instruksi mengenai {task_hint}, saya akan menyusun respons yang sesuai.",
    "Saya perlu mempertimbangkan {task_hint} sebelum memberikan jawaban akhir.",
]


def load_split_dataset(test_size: float = 0.05, seed: int = 1010):
    """Memuat dataset mentah dari HuggingFace, lalu membagi train/val.

    Catatan: dataset ini hanya memiliki kolom ['Unnamed: 0', 'input',
    'output'] -- bukan format Alpaca standar 3-kolom
    (instruction/input/output). Kolom 'input' di sini berisi
    instruksi/pertanyaan itu sendiri, bukan context tambahan. Kolom
    'Unnamed: 0' merupakan artifact index dari CSV export dan di-drop
    karena tidak digunakan.

    Seed di-fix agar split konsisten dan reproducible antar eksperimen
    (supaya perbandingan hyperparameter fair -- data train/val sama).

    Args:
        test_size: Proporsi data yang dialokasikan ke validation split.
        seed: Seed untuk `train_test_split`, fixed supaya split sama
            persis di semua eksperimen hyperparameter.

    Returns:
        Tuple `(train_dataset, val_dataset)`, keduanya `datasets.Dataset`
        dengan kolom `['input', 'output']` (kolom index CSV sudah di-drop).
    """
    dataset = load_dataset(DATASET_NAME, split="train")
    split = dataset.train_test_split(test_size=test_size, seed=seed)

    train_dataset = split["train"].remove_columns(["Unnamed: 0"])
    val_dataset = split["test"].remove_columns(["Unnamed: 0"])

    return train_dataset, val_dataset


def build_formatting_func(tokenizer, system_prompt: str):
    """Mengembalikan formatting function untuk dataset.map(batched=True).

    Dipisah menjadi factory function (bukan formatting_func langsung)
    karena tokenizer dan system_prompt berbeda-beda tergantung
    model/eksperimen yang sedang berjalan -- sehingga tiap notebook
    tinggal memanggil dengan tokenizer dan system_prompt masing-masing.

    Args:
        tokenizer: Tokenizer Unsloth/HF dengan chat template terpasang
            (dipakai untuk `apply_chat_template`).
        system_prompt: System prompt yang disisipkan ke tiap conversation.

    Returns:
        Fungsi `formatting_prompts_func(examples) -> dict` yang menerima
        batch dari `Dataset.map(batched=True)` dan mengembalikan dict
        dengan kolom `"text"` berisi hasil chat-template.
    """

    def formatting_prompts_func(examples):
        user_inputs = examples["input"]
        outputs = examples["output"]
        texts = []

        for user_input, output in zip(user_inputs, outputs):
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input.strip()},
                {"role": "assistant", "content": output.strip()},
            ]
            text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

        return {"text": texts}

    return formatting_prompts_func


def prepare_datasets(
    tokenizer, system_prompt: str, test_size: float = 0.05, seed: int = 1010
):
    """Helper satu panggilan: memuat, membagi, dan memformat dataset.

    Args:
        tokenizer: Tokenizer Unsloth/HF dengan chat template terpasang.
        system_prompt: System prompt yang disisipkan ke tiap conversation.
        test_size: Proporsi data yang dialokasikan ke validation split.
        seed: Seed untuk split, diteruskan ke `load_split_dataset`.

    Returns:
        Tuple `(train_dataset, val_dataset)` -- keduanya sudah memiliki
        kolom `"text"` siap dipakai `SFTTrainer`.
    """
    train_dataset, val_dataset = load_split_dataset(test_size=test_size, seed=seed)
    formatting_func = build_formatting_func(tokenizer, system_prompt)

    train_dataset = train_dataset.map(formatting_func, batched=True)
    val_dataset = val_dataset.map(formatting_func, batched=True)

    return train_dataset, val_dataset


def extract_task_hint(instruction: str, max_words: int = 6) -> str:
    """Mengambil ringkasan singkat dari instruksi untuk placeholder <think>.

    Heuristik: mengabaikan kata perintah umum di depan, mengambil sisa
    kalimat pendek.

    Args:
        instruction: Teks instruksi/pertanyaan mentah (kolom `input`).
        max_words: Jumlah kata maksimum yang diambil dari klausa pertama.

    Returns:
        Ringkasan singkat lowercase, atau `"permintaan pengguna"` sebagai
        fallback apabila instruksi kosong setelah diproses.
    """
    instruction = instruction.strip()
    # Ambil klausa pertama sebelum newline
    first_line = instruction.split("\n")[0]
    words = first_line.split()
    hint = " ".join(words[:max_words]).rstrip(".,:;")
    return hint.lower() if hint else "permintaan pengguna"


def build_coldstart_example(
    row: dict, tokenizer, system_prompt: str, rng: random.Random
) -> dict:
    """Membangun satu contoh cold-start dengan placeholder reasoning `<think>`.

    Konten reasoning diisi dari template acak (`THINK_TEMPLATES`) yang
    diformat menggunakan `task_hint` hasil ekstraksi dari instruksi --
    bukan reasoning asli, hanya placeholder agar model mempelajari pola
    format `<think>...</think>{jawaban}` sebelum memasuki tahap GRPO.

    Args:
        row: Satu baris dataset dengan kolom `input` dan `output`.
        tokenizer: Tokenizer dengan chat template terpasang.
        system_prompt: System prompt yang disisipkan ke conversation.
        rng: Instance `random.Random` (bukan modul `random` global) agar
            pemilihan template deterministic dan reproducible mengikuti
            seed yang diteruskan dari `build_coldstart_dataset`, tanpa
            membocorkan side effect ke global random state.

    Returns:
        Dict dengan kolom `"text"` berisi hasil chat-template lengkap
        (system + user + assistant dengan `<think>` block).
    """
    task_hint = extract_task_hint(row["input"])
    think_content = rng.choice(THINK_TEMPLATES).format(task_hint=task_hint)

    assistant_content = f"<think>\n{think_content}\n</think>\n{row['output']}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["input"]},
        {"role": "assistant", "content": assistant_content},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


def build_coldstart_dataset(
    base_dataset, tokenizer, system_prompt: str, n_samples: int = 300, seed: int = 1010
) -> Dataset:
    """Membangun subset cold-start dari `base_dataset` untuk SFT awal GRPO.

    Seed dipakai untuk dua hal yang sebelumnya rawan tercampur: shuffle
    subset dataset (`base_dataset.shuffle`) dan pemilihan template
    `<think>` per baris (`random.choice`). Keduanya kini di-drive dari
    seed yang sama melalui instance `random.Random` lokal, sehingga
    seluruh proses fully reproducible tanpa menyentuh global random
    state.

    Args:
        base_dataset: Dataset sumber (biasanya `train_dataset` dari
            `load_split_dataset`) yang akan di-subset.
        tokenizer: Tokenizer dengan chat template terpasang.
        system_prompt: System prompt yang disisipkan ke tiap conversation.
        n_samples: Jumlah sampel cold-start yang diambil, di-clip ke
            `len(base_dataset)` apabila dataset lebih kecil.
        seed: Seed untuk shuffle dataset maupun pemilihan template
            `<think>` -- fixed supaya cold-start set reproducible.

    Returns:
        `Dataset` hasil map, kolom lama sudah dibuang, hanya tersisa
        kolom `"text"` siap dipakai SFTTrainer.
    """
    rng = random.Random(seed)
    subset = base_dataset.shuffle(seed=seed).select(
        range(min(n_samples, len(base_dataset)))
    )
    coldstart = subset.map(
        lambda row: build_coldstart_example(row, tokenizer, system_prompt, rng),
        remove_columns=subset.column_names,
    )
    return coldstart
