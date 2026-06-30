import os
import re
import json
import logging
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from groq import Groq
from PIL import Image
import cv2
import io
import base64

# ─── Setup ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

CORS(app, resources={r"/api/*": {"origins": "*"}})

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["60 per minute"],
    storage_uri="memory://"
)

# ─── Groq Client ──────────────────────────────────────────────────────────────

groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

# ─── Health Check ─────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "SmartGrade API",
        "version": "2.0.0",
        "endpoints": ["/api/scan", "/api/grade-text", "/api/health"]
    }), 200


@app.route("/api/health", methods=["GET"])
def api_health():
    groq_status = "ok" if os.environ.get("GROQ_API_KEY") else "missing"
    return jsonify({
        "status": "ok",
        "groq": groq_status
    }), 200


# ─── OMR Scan (Pilihan Ganda) ──────────────────────────────────────────────────

def preprocess_image(image_bytes):
    """Konversi bytes ke grayscale numpy array untuk OpenCV."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    return gray


def detect_bubbles(gray_img, num_questions, num_choices):
    """
    Deteksi bubble yang diarsir pada lembar jawaban.
    Mengembalikan dict {nomor_soal: jawaban_huruf}
    """
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contour berbentuk lingkaran
    bubbles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200 or area > 5000:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if circularity > 0.7:
            x, y, w, h = cv2.boundingRect(cnt)
            bubbles.append((x, y, w, h, cnt))

    if not bubbles:
        return {}

    # Urutkan bubble berdasarkan posisi Y (baris = soal), X (kolom = pilihan)
    bubbles.sort(key=lambda b: (b[1] // 30, b[0]))

    choices = "ABCDE"
    answers = {}
    row_groups = {}

    for (x, y, w, h, cnt) in bubbles:
        row_key = y // 30
        if row_key not in row_groups:
            row_groups[row_key] = []
        row_groups[row_key].append((x, y, w, h, cnt))

    sorted_rows = sorted(row_groups.keys())

    for q_idx, row_key in enumerate(sorted_rows[:num_questions]):
        row_bubbles = sorted(row_groups[row_key], key=lambda b: b[0])
        max_fill = -1
        best_choice = None

        for c_idx, (x, y, w, h, cnt) in enumerate(row_bubbles[:num_choices]):
            mask = np.zeros(gray_img.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_val = cv2.mean(gray_img, mask=mask)[0]
            fill = 255 - mean_val  # semakin gelap = semakin terisi

            if fill > max_fill:
                max_fill = fill
                best_choice = choices[c_idx] if c_idx < len(choices) else str(c_idx + 1)

        if best_choice and max_fill > 50:
            answers[q_idx + 1] = best_choice

    return answers


@app.route("/api/scan", methods=["POST"])
@limiter.limit("30 per minute")
def scan():
    """
    Endpoint OMR untuk koreksi pilihan ganda.
    Menerima: image (file), answer_key (JSON string), num_choices (int)
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "File gambar tidak ditemukan"}), 400

        image_file = request.files["image"]
        answer_key_raw = request.form.get("answer_key", "{}")
        num_choices = int(request.form.get("num_choices", 5))

        # Parse kunci jawaban
        try:
            answer_key = json.loads(answer_key_raw)
            # Normalisasi key ke integer
            answer_key = {int(k): v.upper().strip() for k, v in answer_key.items()}
        except (json.JSONDecodeError, ValueError) as e:
            return jsonify({"error": f"Format kunci jawaban tidak valid: {str(e)}"}), 400

        image_bytes = image_file.read()
        gray_img = preprocess_image(image_bytes)

        num_questions = len(answer_key)
        student_answers = detect_bubbles(gray_img, num_questions, num_choices)

        # Hitung skor
        results = {}
        correct = 0

        for q_num, correct_ans in answer_key.items():
            student_ans = student_answers.get(q_num, "-")
            is_correct = student_ans == correct_ans
            if is_correct:
                correct += 1
            results[q_num] = {
                "kunci": correct_ans,
                "jawaban": student_ans,
                "benar": is_correct
            }

        total = len(answer_key)
        score = round((correct / total) * 100, 2) if total > 0 else 0

        logger.info(f"OMR scan selesai: {correct}/{total} benar, skor={score}")

        return jsonify({
            "status": "success",
            "skor": score,
            "benar": correct,
            "total": total,
            "detail": results
        }), 200

    except Exception as e:
        logger.error(f"Error di /api/scan: {str(e)}", exc_info=True)
        return jsonify({"error": "Terjadi kesalahan saat memproses gambar"}), 500


# ─── AI Grading (Isian & Esai) ────────────────────────────────────────────────

def build_prompt(question, answer_key, student_answer, item_type, max_score):
    """Buat prompt terstruktur untuk Groq LLaMA 3.3."""

    if item_type == "isian":
        return f"""Kamu adalah guru profesional yang mengoreksi jawaban isian singkat siswa sekolah Indonesia.

Pertanyaan: {question}
Kunci Jawaban: {answer_key}
Jawaban Siswa: {student_answer}
Skor Maksimal: {max_score}

Instruksi:
- Bandingkan jawaban siswa dengan kunci jawaban secara SEMANTIK (makna), bukan hanya pencocokan kata
- Jawaban yang memiliki makna sama atau setara dengan kunci jawaban dianggap BENAR
- Berikan skor antara 0 sampai {max_score}
- Berikan komentar singkat dalam Bahasa Indonesia

Kembalikan HANYA JSON valid ini (tanpa teks lain):
{{"skor": <angka>, "status": "<benar|sebagian benar|salah>", "komentar": "<komentar singkat>"}}"""

    else:  # esai
        return f"""Kamu adalah guru profesional yang menilai esai siswa sekolah Indonesia.

Pertanyaan: {question}
Panduan Jawaban / Kunci: {answer_key}
Jawaban Siswa: {student_answer}
Skor Maksimal: {max_score}

Kriteria Penilaian:
1. Relevansi isi dengan pertanyaan (30%)
2. Kelengkapan dan kedalaman jawaban (30%)
3. Pemahaman konsep (25%)
4. Tata bahasa dan struktur kalimat (15%)

Instruksi:
- Nilai secara objektif dan konstruktif
- Berikan feedback yang membantu siswa berkembang
- Gunakan Bahasa Indonesia yang ramah namun profesional

Kembalikan HANYA JSON valid ini (tanpa teks lain):
{{"skor": <angka>, "aspek": {{"relevansi": <0-30>, "kelengkapan": <0-30>, "pemahaman": <0-25>, "bahasa": <0-15>}}, "feedback": "<feedback konstruktif 2-3 kalimat>"}}"""


def extract_json_from_response(text):
    """Ekstrak JSON dari respons AI meskipun ada teks tambahan."""
    text = text.strip()

    # Coba parse langsung
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Cari blok JSON dalam teks
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


@app.route("/api/grade-text", methods=["POST"])
@limiter.limit("10 per minute")
def grade_text():
    """
    Endpoint AI untuk koreksi isian singkat dan esai.
    Menerima: question, answer_key, student_answer, type, max_score
    """
    try:
        data = request.get_json(force=True)

        if not data:
            return jsonify({"error": "Body request tidak valid"}), 400

        question = data.get("question", "").strip()
        answer_key = data.get("answer_key", "").strip()
        student_answer = data.get("student_answer", "").strip()
        item_type = data.get("type", "isian").strip().lower()
        max_score = int(data.get("max_score", 100))

        # Validasi input
        if not question:
            return jsonify({"error": "Pertanyaan tidak boleh kosong"}), 400
        if not answer_key:
            return jsonify({"error": "Kunci jawaban tidak boleh kosong"}), 400
        if not student_answer:
            return jsonify({
                "skor": 0,
                "status": "tidak dijawab",
                "komentar": "Siswa tidak memberikan jawaban.",
                "feedback": "Siswa tidak memberikan jawaban."
            }), 200
        if item_type not in ["isian", "esai"]:
            return jsonify({"error": "Tipe harus 'isian' atau 'esai'"}), 400

        prompt = build_prompt(question, answer_key, student_answer, item_type, max_score)

        logger.info(f"Groq grading: type={item_type}, max_score={max_score}")

        # Panggil Groq API
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Kamu adalah sistem penilaian otomatis. Selalu kembalikan respons dalam format JSON valid tanpa teks tambahan apapun."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=512,
        )

        raw_response = chat_completion.choices[0].message.content
        logger.info(f"Groq raw response: {raw_response[:200]}")

        result = extract_json_from_response(raw_response)

        if not result:
            logger.error(f"Gagal parse JSON dari Groq: {raw_response}")
            return jsonify({"error": "AI mengembalikan format tidak valid, coba lagi"}), 500

        # Normalisasi skor
        ai_score = result.get("skor", 0)
        normalized_score = min(max(float(ai_score), 0), max_score)
        result["skor"] = round(normalized_score, 1)
        result["max_score"] = max_score

        logger.info(f"Grading selesai: skor={result['skor']}/{max_score}")

        return jsonify({
            "status": "success",
            **result
        }), 200

    except Exception as e:
        logger.error(f"Error di /api/grade-text: {str(e)}", exc_info=True)
        return jsonify({"error": "Terjadi kesalahan saat koreksi AI"}), 500


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
