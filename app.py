import os
import re
import json
import base64
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

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ── App Init ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["60 per minute"],
    storage_uri="memory://"
)

# ── Groq Client ───────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ── Health Check ─────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "SmartGrade AI Backend",
        "groq_ready": groq_client is not None
    }), 200


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "status": "ok",
        "groq_ready": groq_client is not None
    }), 200


# ── OMR Scan Endpoint ─────────────────────────────────────────────────────────
@app.route("/api/scan", methods=["POST"])
@limiter.limit("10 per minute")
def scan_omr():
    try:
        data = request.get_json(force=True)

        if not data or "image" not in data:
            return jsonify({"error": "Field 'image' wajib diisi (base64)"}), 400

        answer_key = data.get("answer_key", [])
        if not answer_key:
            return jsonify({"error": "Field 'answer_key' wajib diisi"}), 400

        # Decode base64 image
        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Gagal memproses gambar"}), 400

        # Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Deteksi lingkaran
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=30
        )

        detected_answers = []
        score = 0
        total = len(answer_key)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for i, key in enumerate(answer_key):
                detected_answers.append({
                    "no": i + 1,
                    "kunci": key,
                    "jawaban": "A",   # placeholder — sesuaikan dengan logika OMR
                    "benar": False
                })
        else:
            detected_answers = [
                {
                    "no": i + 1,
                    "kunci": key,
                    "jawaban": "-",
                    "benar": False
                }
                for i, key in enumerate(answer_key)
            ]

        return jsonify({
            "success": True,
            "total_soal": total,
            "skor": score,
            "nilai": round((score / total) * 100, 2) if total > 0 else 0,
            "detail": detected_answers
        }), 200

    except Exception as e:
        logger.error(f"[scan_omr] Error: {e}")
        return jsonify({"error": "Gagal memproses scan OMR", "detail": str(e)}), 500


# ── AI Grading Endpoint ───────────────────────────────────────────────────────
@app.route("/api/grade-text", methods=["POST"])
@limiter.limit("5 per minute")
def grade_text():
    try:
        if not groq_client:
            return jsonify({"error": "GROQ_API_KEY belum dikonfigurasi"}), 503

        data = request.get_json(force=True)

        if not data:
            return jsonify({"error": "Request body kosong"}), 400

        required_fields = ["question", "student_answer", "reference_answer"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Field '{field}' wajib diisi"}), 400

        question         = str(data["question"]).strip()
        student_answer   = str(data["student_answer"]).strip()
        reference_answer = str(data["reference_answer"]).strip()
        max_score        = int(data.get("max_score", 10))
        question_type    = str(data.get("type", "essay"))

        if not student_answer:
            return jsonify({
                "success": True,
                "score": 0,
                "max_score": max_score,
                "percentage": 0,
                "feedback": "Jawaban kosong.",
                "criteria": {}
            }), 200

        # System prompt
        system_prompt = f"""Kamu adalah asisten penilaian otomatis untuk guru.
Nilailah jawaban siswa secara objektif berdasarkan kunci jawaban.

Tipe soal: {question_type}
Skor maksimal: {max_score}

Kembalikan HANYA JSON valid dengan format:
{{
  "score": <angka 0 sampai {max_score}>,
  "percentage": <angka 0-100>,
  "feedback": "<umpan balik singkat dalam Bahasa Indonesia>",
  "criteria": {{
    "accuracy": <0-100>,
    "completeness": <0-100>,
    "clarity": <0-100>
  }}
}}

Jangan tambahkan teks apapun di luar JSON."""

        user_prompt = f"""Pertanyaan: {question}

Kunci Jawaban: {reference_answer}

Jawaban Siswa: {student_answer}"""

        # Call Groq API
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=512,
            timeout=60
        )

        raw_response = completion.choices[0].message.content.strip()
        logger.info(f"[grade_text] Groq response: {raw_response[:200]}")

        # Parse JSON dari response
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if not json_match:
            raise ValueError("Tidak ditemukan JSON dalam response AI")

        result = json.loads(json_match.group())

        # Validasi dan normalisasi
        score      = min(max(float(result.get("score", 0)), 0), max_score)
        percentage = min(max(float(result.get("percentage", 0)), 0), 100)
        feedback   = str(result.get("feedback", ""))
        criteria   = result.get("criteria", {})

        return jsonify({
            "success":    True,
            "score":      round(score, 2),
            "max_score":  max_score,
            "percentage": round(percentage, 2),
            "feedback":   feedback,
            "criteria":   criteria
        }), 200

    except json.JSONDecodeError as e:
        logger.error(f"[grade_text] JSON parse error: {e}")
        return jsonify({"error": "Gagal parse response AI", "detail": str(e)}), 500

    except Exception as e:
        logger.error(f"[grade_text] Error: {e}")
        return jsonify({"error": "Gagal memproses penilaian", "detail": str(e)}), 500


# ── Batch Grading ─────────────────────────────────────────────────────────────
@app.route("/api/grade-batch", methods=["POST"])
@limiter.limit("3 per minute")
def grade_batch():
    try:
        if not groq_client:
            return jsonify({"error": "GROQ_API_KEY belum dikonfigurasi"}), 503

        data = request.get_json(force=True)
        items = data.get("items", [])

        if not items:
            return jsonify({"error": "Field 'items' wajib diisi dan tidak boleh kosong"}), 400

        if len(items) > 10:
            return jsonify({"error": "Maksimal 10 item per batch"}), 400

        results = []
        for item in items:
            try:
                question         = str(item.get("question", ""))
                student_answer   = str(item.get("student_answer", ""))
                reference_answer = str(item.get("reference_answer", ""))
                max_score        = int(item.get("max_score", 10))

                if not student_answer.strip():
                    results.append({
                        "success":    True,
                        "score":      0,
                        "max_score":  max_score,
                        "percentage": 0,
                        "feedback":   "Jawaban kosong."
                    })
                    continue

                completion = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": f"Nilai jawaban siswa. Kembalikan JSON: {{\"score\": <0-{max_score}>, \"percentage\": <0-100>, \"feedback\": \"<string>\"}}"
                        },
                        {
                            "role": "user",
                            "content": f"Soal: {question}\nKunci: {reference_answer}\nJawaban: {student_answer}"
                        }
                    ],
                    temperature=0.1,
                    max_tokens=256,
                    timeout=60
                )

                raw = completion.choices[0].message.content.strip()
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                parsed = json.loads(match.group()) if match else {}

                results.append({
                    "success":    True,
                    "score":      min(max(float(parsed.get("score", 0)), 0), max_score),
                    "max_score":  max_score,
                    "percentage": min(max(float(parsed.get("percentage", 0)), 0), 100),
                    "feedback":   str(parsed.get("feedback", ""))
                })

            except Exception as e:
                logger.error(f"[grade_batch] Item error: {e}")
                results.append({
                    "success": False,
                    "error":   str(e),
                    "score":   0,
                    "max_score": item.get("max_score", 10)
                })

        return jsonify({"success": True, "results": results}), 200

    except Exception as e:
        logger.error(f"[grade_batch] Error: {e}")
        return jsonify({"error": "Gagal memproses batch", "detail": str(e)}), 500


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
