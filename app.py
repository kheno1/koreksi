import os
import base64
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from groq import Groq

app = Flask(__name__)
CORS(app)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["60 per minute"]
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ─────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────
@app.route("/")
def index():
    return jsonify({"status": "SmartGrade API is running", "version": "2.0"})


# ─────────────────────────────────────────
# EXTRACT ANSWERS FROM IMAGE (Vision AI)
# ─────────────────────────────────────────
@app.route("/api/extract-answers", methods=["POST"])
@limiter.limit("30 per minute")
def extract_answers():
    """
    Menerima gambar scan lembar jawaban siswa,
    AI membaca dan mengekstrak semua jawaban secara otomatis.
    """
    data = request.get_json()
    
    if not data or "image" not in data:
        return jsonify({"error": "Image data required"}), 400
    
    mode = data.get("mode", "pg")  # pg | pg_isian | pg_isian_essay
    image_base64 = data.get("image", "")
    
    # Bersihkan prefix base64 jika ada
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
    
    # Buat prompt sesuai mode
    if mode == "pg":
        extraction_prompt = """
Kamu adalah sistem OCR untuk lembar jawaban ujian sekolah Indonesia.

Baca gambar lembar jawaban siswa ini dengan SANGAT TELITI.

Ekstrak semua jawaban pilihan ganda siswa.
Format output HARUS berupa JSON valid seperti ini:
{
  "nama_siswa": "nama jika tertulis, atau null",
  "kelas": "kelas jika tertulis, atau null", 
  "pilihan_ganda": {
    "1": "A",
    "2": "C",
    "3": "B",
    "4": "D",
    "5": "A"
  },
  "total_soal_terdeteksi": 5,
  "catatan": "catatan jika ada jawaban yang tidak terbaca jelas"
}

Aturan:
- Jawaban PG hanya boleh A, B, C, D, atau E
- Jika tidak terbaca tulis "?"
- Hanya output JSON, tidak ada teks lain
"""

    elif mode == "pg_isian":
        extraction_prompt = """
Kamu adalah sistem OCR untuk lembar jawaban ujian sekolah Indonesia.

Baca gambar lembar jawaban siswa ini dengan SANGAT TELITI.

Ekstrak semua jawaban pilihan ganda DAN isian singkat siswa.
Format output HARUS berupa JSON valid seperti ini:
{
  "nama_siswa": "nama jika tertulis, atau null",
  "kelas": "kelas jika tertulis, atau null",
  "pilihan_ganda": {
    "1": "A",
    "2": "C"
  },
  "isian_singkat": {
    "1": "jawaban siswa untuk isian 1",
    "2": "jawaban siswa untuk isian 2"
  },
  "total_pg_terdeteksi": 2,
  "total_isian_terdeteksi": 2,
  "catatan": "catatan jika ada tulisan tidak terbaca"
}

Aturan:
- Jawaban PG hanya A, B, C, D, atau E
- Isian: tulis persis apa yang ditulis siswa
- Jika tidak terbaca tulis "?"
- Hanya output JSON, tidak ada teks lain
"""

    else:  # pg_isian_essay
        extraction_prompt = """
Kamu adalah sistem OCR untuk lembar jawaban ujian sekolah Indonesia.

Baca gambar lembar jawaban siswa ini dengan SANGAT TELITI dan LENGKAP.

Ekstrak semua jawaban: pilihan ganda, isian singkat, DAN esai panjang.
Format output HARUS berupa JSON valid seperti ini:
{
  "nama_siswa": "nama jika tertulis, atau null",
  "kelas": "kelas jika tertulis, atau null",
  "pilihan_ganda": {
    "1": "A",
    "2": "C"
  },
  "isian_singkat": {
    "1": "jawaban isian 1",
    "2": "jawaban isian 2"
  },
  "essay": {
    "1": "Teks lengkap esai siswa nomor 1 ditulis selengkap mungkin sesuai yang tertulis di kertas",
    "2": "Teks lengkap esai siswa nomor 2"
  },
  "total_pg_terdeteksi": 2,
  "total_isian_terdeteksi": 2,
  "total_essay_terdeteksi": 2,
  "catatan": "catatan jika ada tulisan tidak terbaca atau gambar kurang jelas"
}

Aturan PENTING:
- Jawaban PG hanya A, B, C, D, atau E
- Isian: tulis persis apa yang ditulis siswa
- Esai: tulis SELENGKAP mungkin, jangan potong
- Jika tidak terbaca tulis "?"
- Hanya output JSON, tidak ada teks lain
"""

    try:
        # Gunakan Groq Vision (llama-4-scout atau meta-llama/llama-4-maverick)
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": extraction_prompt
                        }
                    ]
                }
            ],
            temperature=0.1,  # Rendah untuk akurasi OCR
            max_tokens=2000
        )
        
        raw_text = response.choices[0].message.content.strip()
        
        # Parse JSON dari response
        # Bersihkan jika ada markdown code block
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()
        
        extracted_data = json.loads(raw_text)
        
        return jsonify({
            "success": True,
            "data": extracted_data,
            "mode": mode
        })
        
    except json.JSONDecodeError as e:
        return jsonify({
            "success": False,
            "error": "Gagal parse JSON dari AI",
            "raw_response": raw_text if 'raw_text' in locals() else "",
            "detail": str(e)
        }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ─────────────────────────────────────────
# GRADE TEXT (AI Koreksi Isian & Esai)
# ─────────────────────────────────────────
@app.route("/api/grade-text", methods=["POST"])
@limiter.limit("30 per minute")
def grade_text():
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    question    = data.get("question", "")
    student_ans = data.get("student_answer", "")
    key_ans     = data.get("key_answer", "")
    q_type      = data.get("type", "isian")  # isian | essay
    max_score   = data.get("max_score", 10)
    
    if q_type == "essay":
        system_prompt = f"""
Kamu adalah guru pengoreksi esai yang adil dan teliti untuk sekolah di Indonesia.

Koreksi jawaban esai siswa berdasarkan kunci jawaban/poin-poin yang harus ada.

Soal: {question}
Kunci Jawaban / Poin Penting: {key_ans}
Jawaban Siswa: {student_ans}
Skor Maksimal: {max_score}

Berikan penilaian dalam format JSON:
{{
  "skor": <angka 0 sampai {max_score}>,
  "persentase": <0-100>,
  "komentar": "komentar konstruktif dalam bahasa Indonesia",
  "kelebihan": "apa yang sudah benar dari jawaban siswa",
  "kekurangan": "apa yang kurang atau salah",
  "poin_terpenuhi": ["poin1", "poin2"],
  "poin_kurang": ["poin3"]
}}
"""
    else:  # isian singkat
        system_prompt = f"""
Kamu adalah guru pengoreksi isian singkat yang adil untuk sekolah di Indonesia.

Koreksi jawaban isian singkat siswa. Toleransi typo kecil diperbolehkan.

Soal: {question}
Kunci Jawaban: {key_ans}
Jawaban Siswa: {student_ans}
Skor Maksimal: {max_score}

Berikan penilaian dalam format JSON:
{{
  "skor": <angka 0 sampai {max_score}>,
  "benar": <true atau false>,
  "persentase": <0-100>,
  "komentar": "komentar singkat dalam bahasa Indonesia",
  "kunci_jawaban": "{key_ans}"
}}
"""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": system_prompt}],
            temperature=0.2,
            max_tokens=800
        )
        
        raw = response.choices[0].message.content.strip()
        
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        
        result = json.loads(raw)
        return jsonify({"success": True, "result": result})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ─────────────────────────────────────────
# GRADE BATCH (Koreksi Banyak Soal Sekaligus)
# ─────────────────────────────────────────
@app.route("/api/grade-batch", methods=["POST"])
@limiter.limit("20 per minute")
def grade_batch():
    data = request.get_json()
    
    if not data or "items" not in data:
        return jsonify({"error": "Items array required"}), 400
    
    items = data["items"][:15]  # Max 15 soal
    results = []
    total_score = 0
    max_total = 0
    
    for item in items:
        q_type      = item.get("type", "isian")
        question    = item.get("question", f"Soal nomor {item.get('number', '?')}")
        student_ans = item.get("student_answer", "")
        key_ans     = item.get("key_answer", "")
        max_score   = item.get("max_score", 10)
        number      = item.get("number", 0)
        
        if q_type == "pg":
            # Koreksi PG langsung tanpa AI
            benar = str(student_ans).strip().upper() == str(key_ans).strip().upper()
            skor = max_score if benar else 0
            results.append({
                "number": number,
                "type": "pg",
                "student_answer": student_ans,
                "key_answer": key_ans,
                "benar": benar,
                "skor": skor,
                "max_score": max_score
            })
            total_score += skor
            max_total += max_score
            continue
        
        # Isian / Esai → pakai AI
        if q_type == "essay":
            prompt = f"""
Koreksi jawaban esai. Soal: {question}
Kunci: {key_ans}
Jawaban Siswa: {student_ans}
Skor Maks: {max_score}

Output JSON: {{"skor": <angka>, "persentase": <0-100>, "komentar": "..."}}
"""
        else:
            prompt = f"""
Koreksi isian singkat. Soal: {question}
Kunci: {key_ans}  
Jawaban Siswa: {student_ans}
Skor Maks: {max_score}

Toleransi typo kecil. Output JSON: {{"skor": <angka>, "benar": <true/false>, "komentar": "..."}}
"""
        
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            raw = response.choices[0].message.content.strip()
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            
            graded = json.loads(raw)
            graded["number"] = number
            graded["type"] = q_type
            graded["student_answer"] = student_ans
            graded["key_answer"] = key_ans
            graded["max_score"] = max_score
            
            results.append(graded)
            total_score += graded.get("skor", 0)
            max_total += max_score
            
        except Exception as e:
            results.append({
                "number": number,
                "type": q_type,
                "error": str(e),
                "skor": 0,
                "max_score": max_score
            })
            max_total += max_score
    
    final_percentage = (total_score / max_total * 100) if max_total > 0 else 0
    
    return jsonify({
        "success": True,
        "results": results,
        "summary": {
            "total_score": round(total_score, 2),
            "max_total": max_total,
            "percentage": round(final_percentage, 2),
            "grade": get_grade(final_percentage)
        }
    })


def get_grade(percentage):
    if percentage >= 90: return "A"
    elif percentage >= 80: return "B"
    elif percentage >= 70: return "C"
    elif percentage >= 60: return "D"
    else: return "E"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
