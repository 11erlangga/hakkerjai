"""
reward_functions.py

Kumpulan reward function custom untuk GRPOTrainer (TRL + Unsloth) pada
tahap GRPO dalam pipeline fine-tuning Legal RAG SLM.

Empat reward function:
1. format_reward_func            -> reward shaping tag <think>...</think>, max +1.0
2. reasoning_length_reward_func  -> reward proporsional panjang isi <think>
3. correctness_reward_func       -> reward berdasarkan ground truth (containment / ROUGE-L)
4. language_reward_func          -> reward berdasarkan bahasa output akhir (ID vs EN)

Semua function mengikuti signature yang diharapkan GRPOTrainer:
    def reward_func(completions, **kwargs) -> list[float]

Catatan implementasi:
- ROUGE_SIMILARITY_THRESHOLD sudah dikalibrasi berdasarkan distribusi
  skor ROUGE-L pada [sumber/metode kalibrasi] -- lihat
  notebooks/02_model_selection_and_reward_tuning.ipynb untuk detail
  proses dan alasan pemilihan nilai.
- language_reward_func hanya menilai bahasa pada FINAL ANSWER (setelah
  </think>), bukan seluruh teks termasuk isi reasoning. Ini keputusan
  desain eksplisit: isi reasoning dalam <think> dianggap ruang bebas
  bagi model untuk bernalar, sedangkan yang perlu dijamin berbahasa
  Indonesia adalah jawaban yang benar-benar dibaca pengguna. Trade-off:
  model bisa saja "curang" bernalar dalam bahasa Inggris tanpa kena
  penalti, selama jawaban akhirnya tetap berbahasa Indonesia.
"""

from langdetect import DetectorFactory, LangDetectException, detect
from rouge_score import rouge_scorer

ROUGE_SIMILARITY_THRESHOLD = 0.2

_rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

# langdetect menggunakan random projection secara internal sehingga
# hasil detect() bisa berbeda antar pemanggilan untuk input yang sama.
# Seed di-fix agar reward dari language_reward_func deterministic --
# tanpa ini, reward GRPO bisa berubah-ubah untuk completion yang
# identik, menambah noise yang bukan berasal dari model.
DetectorFactory.seed = 1010


def extract_final_answer(text: str) -> str:
    """Mengambil jawaban akhir dari teks completion.

    Jawaban akhir didefinisikan sebagai seluruh teks setelah `</think>`
    yang PERTAMA. Apabila tidak ada `</think>` sama sekali, seluruh teks
    dianggap sebagai jawaban (model belum/tidak memakai format
    reasoning, tapi tetap dinilai).

    Args:
        text: Teks completion mentah dari model.

    Returns:
        Substring setelah `</think>` pertama, sudah di-strip whitespace.
    """
    close_idx = text.find("</think>")
    if close_idx == -1:
        return text.strip()
    return text[close_idx + len("</think>") :].strip()


def _get_completion_text(completion) -> str:
    """Mengekstrak string content dari satu elemen `completions` GRPOTrainer.

    Args:
        completion: Satu elemen dari list `completions`, dengan format
            `[{"role": ..., "content": "..."}]`.

    Returns:
        Isi field `"content"` dari elemen pertama.
    """
    return completion[0]["content"]


def format_reward_func(completions, **kwargs) -> list[float]:
    """Memberi reward shaping untuk format `<think>...</think>` sebelum jawaban.

    Skema (maksimum +1.0):
    - Format sempurna (dimulai `<think>`, ditutup dengan benar, diikuti
      jawaban akhir, masing-masing tag muncul tepat satu kali) -> +1.0
      (eksklusif, tidak ditumpuk dengan reward parsial di bawah).
    - Apabila tidak sempurna, reward parsial (additive):
        - Dimulai dengan `<think>` -> +0.2
        - Ada `</think>` di suatu tempat -> +0.3
    - Penalti -0.5 apabila `<think>` atau `</think>` muncul lebih dari
      satu kali (halusinasi), diterapkan independen dari skema di atas.

    Args:
        completions: List completion dari GRPOTrainer.
        **kwargs: Argumen tambahan dari GRPOTrainer (tidak dipakai).

    Returns:
        List reward (float) sejajar dengan `completions`.
    """
    rewards = []
    for completion in completions:
        text = _get_completion_text(completion)
        reward = 0.0

        open_count = text.count("<think>")
        close_count = text.count("</think>")
        starts_with_think = text.strip().startswith("<think>")
        has_closing = close_count >= 1

        followed_by_answer = False
        if has_closing:
            after_close = text.split("</think>", 1)[1].strip()
            followed_by_answer = len(after_close) > 0

        is_perfect = (
            starts_with_think
            and has_closing
            and followed_by_answer
            and open_count == 1
            and close_count == 1
        )

        if is_perfect:
            reward = 1.0
        else:
            if starts_with_think:
                reward += 0.2
            if has_closing:
                reward += 0.3

        if open_count > 1 or close_count > 1:
            reward -= 0.5

        rewards.append(reward)

    return rewards


def reasoning_length_reward_func(completions, **kwargs) -> list[float]:
    """Memberi reward proporsional terhadap panjang isi `<think>...</think>`.

    Toleran apabila reasoning terpotong oleh token limit (artinya
    `<think>` ada tapi `</think>` belum sempat muncul karena
    `max_completion_length` terpotong).

    Skema:
    - Tidak ada tag / isi kosong -> 0.0
    - <50 karakter               -> 0.2
    - 50-199 karakter            -> 0.5
    - >=200 karakter             -> 1.0

    Catatan: hanya menghitung pasangan `<think>...</think>` PERTAMA.
    Kasus `<think>` muncul berkali-kali (halusinasi) sudah dihukum
    terpisah di `format_reward_func`, sehingga di sini tidak perlu
    di-double-handle.

    Args:
        completions: List completion dari GRPOTrainer.
        **kwargs: Argumen tambahan dari GRPOTrainer (tidak dipakai).

    Returns:
        List reward (float) sejajar dengan `completions`.
    """
    rewards = []
    for completion in completions:
        text = _get_completion_text(completion)

        open_idx = text.find("<think>")
        if open_idx == -1:
            rewards.append(0.0)
            continue

        content_start = open_idx + len("<think>")
        close_idx = text.find("</think>", content_start)

        if close_idx == -1:
            # <think> ada tapi </think> belum muncul -> kemungkinan
            # terpotong token limit. Toleran: anggap semua sisa teks
            # setelah <think> sebagai isi reasoning yang belum sempat
            # ditutup.
            reasoning_content = text[content_start:]
        else:
            reasoning_content = text[content_start:close_idx]

        length = len(reasoning_content.strip())

        if length == 0:
            reward = 0.0
        elif length < 50:
            reward = 0.2
        elif length < 200:
            reward = 0.5
        else:
            reward = 1.0

        rewards.append(reward)

    return rewards


def correctness_reward_func(prompts, completions, output, **kwargs) -> list[float]:
    """Memberi reward berdasarkan kecocokan jawaban akhir dengan ground truth.

    Reward +1.0 apabila jawaban akhir:
      (a) MENGANDUNG ground truth sebagai substring, ATAU
      (b) mirip secara ROUGE-L (fmeasure >= ROUGE_SIMILARITY_THRESHOLD)
    Dua kondisi bersifat independen (OR).

    Args:
        prompts: List prompt dari GRPOTrainer (tidak dipakai langsung,
            dipertahankan di signature karena diteruskan otomatis oleh
            GRPOTrainer).
        completions: List completion dari GRPOTrainer.
        output: List ground truth, sejajar dengan `completions`. Nama
            parameter ini HARUS persis `output` -- GRPOTrainer
            meneruskan kolom dataset secara otomatis berdasarkan nama
            kolom, dan kolom ground truth pada dataset Alpaca-GPT4-
            Indonesian bernama `"output"` (lihat `data_utils.py`).
            Mengganti nama parameter ini akan menyebabkan `TypeError`
            karena GRPOTrainer tidak lagi tahu argumen mana yang harus
            diisi dari kolom tersebut.
        **kwargs: Argumen tambahan dari GRPOTrainer (tidak dipakai).

    Returns:
        List reward (float) sejajar dengan `completions`.
    """
    responses = [_get_completion_text(c) for c in completions]
    extracted_answers = [extract_final_answer(r) for r in responses]

    rewards = []
    for pred, gt in zip(extracted_answers, output):
        pred_norm = pred.strip().lower()
        gt_norm = gt.strip().lower()

        if len(pred_norm) == 0 or len(gt_norm) == 0:
            rewards.append(0.0)
            continue

        # Kondisi (a): containment
        contains_gt = gt_norm in pred_norm

        # Kondisi (b): similarity
        score = _rouge_scorer.score(gt_norm, pred_norm)
        rouge_l_f1 = score["rougeL"].fmeasure
        is_similar = rouge_l_f1 >= ROUGE_SIMILARITY_THRESHOLD

        rewards.append(1.0 if (contains_gt or is_similar) else 0.0)

    return rewards


def language_reward_func(completions, **kwargs) -> list[float]:
    """Memberi reward berdasarkan bahasa jawaban akhir (setelah `</think>`).

    Skema:
    - Murni Bahasa Indonesia -> +1.0
    - Tiba-tiba menjawab Bahasa Inggris -> -0.5
    - Bahasa lain / gagal dideteksi / jawaban kosong -> 0.0 (netral,
      hanya dua skenario eksplisit ID dan EN yang mendapat reward
      non-netral)

    Catatan: hanya menilai bahasa pada FINAL ANSWER (setelah
    `</think>`), bukan seluruh teks termasuk isi reasoning -- lihat
    catatan desain di docstring module.

    Args:
        completions: List completion dari GRPOTrainer.
        **kwargs: Argumen tambahan dari GRPOTrainer (tidak dipakai).

    Returns:
        List reward (float) sejajar dengan `completions`.
    """
    responses = [_get_completion_text(c) for c in completions]
    final_answers = [extract_final_answer(r) for r in responses]

    rewards = []
    for answer_text in final_answers:
        text = answer_text.strip()

        if len(text) == 0:
            rewards.append(0.0)
            continue

        try:
            detected_lang = detect(text)
        except LangDetectException:
            rewards.append(0.0)
            continue

        if detected_lang == "en":
            rewards.append(-0.5)
        elif detected_lang == "id":
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards
