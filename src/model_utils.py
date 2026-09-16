"""
model_utils.py

Utilitas untuk memuat model dan tokenizer (Unsloth) serta melakukan
setup PEFT/LoRA. Modul ini dipakai bersama di seluruh notebook
eksperimen SFT.
"""

from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


def load_model_and_tokenizer(
    model_name: str,
    chat_template: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    """Memuat base model (quantized) dan tokenizer, lalu menerapkan chat template.

    Catatan: `load_in_4bit=True` memuat model melalui `BitsAndBytesConfig`
    bawaan Unsloth, yang secara default juga mengaktifkan double
    quantization (`bnb_4bit_use_double_quant=True`) -- sesuai requirement
    QLoRA di brief, meskipun parameter tersebut tidak diset eksplisit di
    sini karena merupakan default internal Unsloth.

    Catatan: nama `chat_template` harus PERSIS cocok dengan salah satu
    key di `unsloth.chat_templates.CHAT_TEMPLATES` -- cek dulu dengan:
        from unsloth.chat_templates import CHAT_TEMPLATES
        print(list(CHAT_TEMPLATES.keys()))
    sebelum mengganti `model_name` ke family lain, karena template harus
    sesuai model (misal Qwen2.5 -> "qwen-2.5", Llama-3.1 -> "llama-3.1").

    Args:
        model_name: Nama/path model dasar di HuggingFace Hub, harus
            didukung Unsloth (Llama, Mistral, Qwen, Gemma, Phi).
        chat_template: Key template pada `CHAT_TEMPLATES`, harus sesuai
            family model yang dipakai.
        max_seq_length: Panjang maksimum sequence (token) yang didukung
            model setelah load.
        load_in_4bit: Jika True, model dimuat dalam quantized 4-bit
            (QLoRA) via Unsloth, termasuk double quantization secara
            default.

    Returns:
        Tuple `(model, tokenizer)` -- model sudah quantized, tokenizer
        sudah dipasangi chat template.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
        dtype=None,
    )

    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    return model, tokenizer


def apply_lora(
    model,
    r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    target_modules: list[str] | None = None,
):
    """Menerapkan adapter LoRA ke model.

    Parameter dibuat menjadi argumen (bukan hardcoded) agar mudah
    dibedakan antar eksperimen hyperparameter (misal eksperimen 1
    memakai r=16, eksperimen 2 mencoba r=32) tanpa duplikasi seluruh
    fungsi. Nilai default `r=16, lora_alpha=16` merupakan hasil
    eksperimen run1 -- lihat progress log untuk detail perbandingan
    loss curve antar eksperimen.

    Args:
        model: Model hasil `load_model_and_tokenizer` (base model
            quantized) yang akan dipasangi adapter LoRA.
        r: Rank adapter LoRA.
        lora_alpha: Faktor scaling LoRA.
        lora_dropout: Dropout rate pada layer LoRA.
        target_modules: Daftar nama modul yang dipasangi adapter. Jika
            None, default mencakup attention (q_proj, k_proj, v_proj,
            o_proj) dan FFN (gate_proj, up_proj, down_proj) sekaligus --
            melebihi requirement minimum brief ("minimal salah satu").

    Returns:
        Model dengan adapter LoRA terpasang, siap dipakai `SFTTrainer`.
    """
    if target_modules is None:
        # Default: mencakup attention (q,k,v,o) + FFN (gate,up,down)
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    model = FastLanguageModel.get_peft_model(
        model,
        target_modules=target_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    return model


def run_coldstart_sft(
    model,
    tokenizer,
    coldstart_dataset,
    output_dir: str = "coldstart_checkpoint",
    max_steps: int = 150,
    seed: int = 1010,
):
    """Menjalankan mini-SFT singkat sebagai tahap cold-start sebelum GRPO.

    Dipakai di atas model yang sudah memiliki adapter LoRA terpasang
    (hasil `apply_lora`, biasanya checkpoint dari run pemenang tahap
    Skilled). `max_steps` sengaja dibuat kecil -- tujuannya hanya
    menanamkan prior format `<think>...</think>` ke model, bukan
    melakukan re-training penuh.

    Args:
        model: Model dengan adapter LoRA terpasang.
        tokenizer: Tokenizer yang sudah dipasangi chat template.
        coldstart_dataset: Dataset hasil `build_coldstart_dataset`,
            dengan kolom `"text"`.
        output_dir: Direktori penyimpanan checkpoint hasil training.
        max_steps: Jumlah step training, sengaja kecil karena tujuannya
            hanya injeksi format, bukan training penuh.
        seed: Seed untuk `SFTConfig`, memastikan run reproducible.

    Returns:
        Instance `SFTTrainer` setelah `trainer.train()` selesai
        dijalankan (checkpoint sudah tersimpan di `output_dir`).
    """
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        seed=seed,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=coldstart_dataset,
        args=sft_config,
    )

    trainer.train()
    return trainer
