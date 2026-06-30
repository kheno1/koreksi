import os
import cv2
import numpy as np
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)

# ── Groq Client ────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 1: OMR - Pilihan Ganda (existing)
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/scan', methods=['POST'])
def scan_omr():
    """Proses lembar jawaban OMR untuk soal pilihan ganda."""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file       = request.files['image']
        answer_key = json.loads(request.form.get('answer_key', '[]'))
        num_q      = int(request.form.get('num_questions', len(answer_key)))

        # Baca gambar
        img_bytes = file.read()
        nparr     = np.frombuffer(img_bytes, np.uint8)
        img       = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        # Proses OMR
        detected = process_omr(img, num_q)

        # Hitung skor
        score, details = calculate_score(detected, answer_key)

        return jsonify({
            "success"  : True,
            "detected" : detected,
            "score"    : score,
            "details"  : details,
            "total"    : num_q
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 2: AI Grade - Isian Singkat & Esai (NEW)
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/grade-text', methods=['POST'])
def grade_text():
    """
    Koreksi isian singkat dan esai menggunakan Groq LLaMA 3.3.
    
    Request body:
    {
        "items": [
            {
                "id"            : "q1",
                "type"          : "isian" | "esai",
                "question"      : "Apa ibukota Indonesia?",
                "answer_key"    : "Jakarta",
                "student_answer": "jakarta",
                "max_score"     : 10
            }
        ],
        "student_name"  : "Budi Santoso",
        "subject"       : "IPS"
    }
    """
    try:
        data         = request.json
        items        = data.get('items', [])
        student_name = data.get('student_name', 'Siswa')
        subject      = data.get('subject', 'Umum')

        if not items:
            return jsonify({"error": "No items provided"}), 400

        results = []

        for item in items:
            item_type   = item.get('type', 'isian')
            question    = item.get('question', '')
            answer_key  = item.get('answer_key', '')
            student_ans = item.get('student_answer', '')
            max_score   = item.get('max_score', 10)
            item_id     = item.get('id', '')

            # Lewati jika jawaban kosong
            if not student_ans.strip():
                results.append({
                    "id"      : item_id,
                    "skor"    : 0,
                    "max"     : max_score,
                    "status"  : "tidak dijawab",
                    "feedback": "Siswa tidak memberikan jawaban."
                })
                continue

            # Bangun prompt berdasarkan tipe soal
            prompt = build_prompt(
                item_type, question, answer_key,
                student_ans, max_score, subject
            )

            # Panggil Groq API
            ai_result = call_groq(prompt)

            # Normalisasi skor ke max_score
            if ai_result.get('skor') is not None:
                normalized = round(
                    (ai_result['skor'] / 100) * max_score, 1
                )
                ai_result['skor']     = normalized
                ai_result['max']      = max_score
                ai_result['id']       = item_id
                ai_result['raw_pct']  = ai_result.get('skor_persen', 0)

            results.append(ai_result)

        # Hitung total skor isian/esai
        total_earned = sum(r.get('skor', 0) for r in results)
        total_max    = sum(item.get('max_score', 10) for item in items)

        return jsonify({
            "success"        : True,
            "student_name"   : student_name,
            "results"        : results,
            "total_earned"   : total_earned,
            "total_max"      : total_max,
            "percentage"     : round((total_earned / total_max * 100), 1) if total_max > 0 else 0
        })

    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 3: Health Check
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/health', methods=['GET'])
def health():
    groq_status = "ok"
    try:
        # Test koneksi Groq ringan
        groq_client.models.list()
    except Exception:
        groq_status = "error"

    return jsonify({
        "status"    : "ok",
        "groq"      : groq_status,
        "model"     : "llama-3.3-70b-versatile"
    })


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def build_prompt(item_type, question, answer_key, student_ans, max_score, subject):
    """Bangun prompt yang sesuai untuk setiap tipe soal."""

    base_context = f"""Kamu adalah guru {subject} yang berpengalaman dan sedang mengoreksi ulangan siswa.
Berikan penilaian yang adil, objektif, dan konstruktif dalam Bahasa Indonesia.
PENTING: Jawab HANYA dengan format JSON yang valid, tanpa teks tambahan di luar JSON."""

    if item_type == 'isian':
        return f"""{base_context}

Tipe Soal: Isian Singkat
Pertanyaan: {question}
Kunci Jawaban: {answer_key}
Jawaban Siswa: {student_ans}

Koreksi jawaban siswa. Pertimbangkan sinonim, ejaan berbeda, dan jawaban yang secara makna sama.

Kembalikan JSON dengan format TEPAT ini:
{{
  "skor_persen": <angka 0-100>,
  "status": "<benar|sebagian benar|salah>",
  "feedback": "<komentar singkat 1-2 kalimat>"
}}"""

    else:  # esai
        return f"""{base_context}

Tipe Soal: Esai
Pertanyaan: {question}
Panduan Jawaban / Kunci: {answer_key}
Jawaban Siswa: {student_ans}
Skor Maksimal: {max_score}

Nilai esai berdasarkan 4 aspek:
1. Relevansi (kesesuaian dengan pertanyaan)
2. Kelengkapan (mencakup poin-poin penting)
3. Pemahaman konsep (kedalaman analisis)
4. Bahasa (kejelasan dan tata bahasa)

Kembalikan JSON dengan format TEPAT ini:
{{
  "skor_persen": <angka 0-100>,
  "status": "<sangat baik|baik|cukup|kurang>",
  "aspek": {{
    "relevansi"  : <0-25>,
    "kelengkapan": <0-25>,
    "pemahaman"  : <0-25>,
    "bahasa"     : <0-25>
  }},
  "feedback": "<komentar konstruktif 2-3 kalimat untuk siswa>"
}}"""


def call_groq(prompt: str) -> dict:
    """Panggil Groq API dengan LLaMA 3.3 dan parse hasilnya."""
    try:
        response = groq_client.chat.completions.create(
            model       = "llama-3.3-70b-versatile",
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.2,   # Rendah agar konsisten
            max_tokens  = 512,
            # Paksa output JSON
            response_format={"type": "json_object"}
        )

        raw_text = response.choices[0].message.content.strip()
        result   = json.loads(raw_text)

        # Normalisasi key
        return {
            "skor"       : result.get("skor_persen", 0),
            "skor_persen": result.get("skor_persen", 0),
            "status"     : result.get("status", "tidak diketahui"),
            "feedback"   : result.get("feedback", ""),
            "aspek"      : result.get("aspek", {})
        }

    except json.JSONDecodeError:
        # Fallback: coba ekstrak JSON dari teks
        return extract_json_fallback(raw_text)
    except Exception as e:
        return {
            "skor"    : 0,
            "status"  : "error",
            "feedback": f"Gagal memproses dengan AI: {str(e)}"
        }


def extract_json_fallback(text: str) -> dict:
    """Ekstrak JSON dari respons teks jika parsing langsung gagal."""
    import re
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {"skor": 0, "status": "error", "feedback": "Format respons AI tidak valid."}


def process_omr(img, num_questions: int) -> list:
    """Proses gambar OMR dan deteksi bubble yang diisi."""
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred   = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 500 < area < 5000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect     = w / float(h)
            if 0.7 < aspect < 1.3:
                bubbles.append((x, y, w, h, cnt))

    # Sort berdasarkan posisi Y (baris) lalu X (kolom)
    bubbles.sort(key=lambda b: (b[1] // 30, b[0]))

    detected  = []
    options   = ['A', 'B', 'C', 'D', 'E']
    num_opts  = 4  # Default 4 pilihan

    for q_idx in range(num_questions):
        start   = q_idx * num_opts
        end     = start + num_opts
        row     = bubbles[start:end]

        if not row:
            detected.append(None)
            continue

        # Cari bubble dengan density tertinggi (paling gelap = diisi)
        best_density = -1
        best_opt     = None

        for opt_idx, bubble in enumerate(row):
            x, y, w, h, cnt = bubble
            mask             = np.zeros(thresh.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            filled   = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
            density  = filled / cv2.contourArea(cnt) if cv2.contourArea(cnt) > 0 else 0

            if density > best_density:
                best_density = density
                best_opt     = options[opt_idx] if opt_idx < len(options) else str(opt_idx + 1)

        detected.append(best_opt if best_density > 0.5 else None)

    return detected


def calculate_score(detected: list, answer_key: list) -> tuple:
    """Hitung skor berdasarkan jawaban terdeteksi vs kunci jawaban."""
    correct = 0
    details = []

    for i, (det, key) in enumerate(zip(detected, answer_key)):
        is_correct = str(det).upper() == str(key).upper() if det else False
        if is_correct:
            correct += 1
        details.append({
            "no"        : i + 1,
            "detected"  : det,
            "key"       : key,
            "correct"   : is_correct
        })

    total = len(answer_key)
    score = round((correct / total * 100), 1) if total > 0 else 0

    return score, details


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
