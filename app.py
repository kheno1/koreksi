from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import base64
import json
import re
import os
import time
from functools import wraps

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# =============================================
# RATE LIMITING
# =============================================
request_times = []

def rate_limit(max_requests=10, window=60):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            now = time.time()
            global request_times
            request_times = [t for t in request_times if now - t < window]
            if len(request_times) >= max_requests:
                return jsonify({"error": "Rate limit exceeded. Coba lagi dalam 1 menit."}), 429
            request_times.append(now)
            return f(*args, **kwargs)
        return wrapped
    return decorator


# =============================================
# HELPER: Bersihkan JSON dari response AI
# =============================================
def clean_json_response(text):
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    # Coba parse langsung
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Cari pola JSON dalam teks
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass
    
    return None


# =============================================
# HELPER: Validasi dan normalisasi hasil ekstraksi
# =============================================
def validate_extraction(data):
    # Pastikan nama tidak fallback ke default
    invalid_names = ['', 'Siswa 1', 'Unknown', 'N/A', 'null', 'undefined']
    if not data.get('nama') or data['nama'] in invalid_names:
        data['nama'] = '[Nama tidak terbaca - harap isi manual]'
    
    # Normalisasi jawaban PG: pastikan huruf kapital
    if 'jawaban_pg' in data and isinstance(data['jawaban_pg'], list):
        normalized = []
        for j in data['jawaban_pg']:
            if j and str(j).strip():
                val = str(j).strip().upper()
                # Ambil hanya huruf pertama jika ada teks panjang
                val = val[0] if val else '?'
                normalized.append(val)
            else:
                normalized.append('?')
        data['jawaban_pg'] = normalized
    
    # Pastikan semua field ada
    data.setdefault('kelas', '')
    data.setdefault('mapel', '')
    data.setdefault('tanggal', '')
    data.setdefault('jawaban_pg', [])
    data.setdefault('jawaban_isian', [])
    data.setdefault('jawaban_essay', [])
    data.setdefault('catatan_keterbacaan', '')
    
    return data


# =============================================
# ENDPOINT 1: Ekstrak Jawaban dari Gambar Scan
# =============================================
@app.route('/api/extract-answers', methods=['POST'])
@rate_limit(max_requests=20, window=60)
def extract_answers():
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "Field 'image' (base64) diperlukan"}), 400
        
        image_base64 = data['image']
        exam_mode = data.get('mode', 'full')
        jumlah_pg = data.get('jumlah_pg', 0)
        jumlah_isian = data.get('jumlah_isian', 0)
        jumlah_essay = data.get('jumlah_essay', 0)
        
        # Bangun instruksi berdasarkan mode ujian
        mode_instruksi = ""
        if exam_mode == "pg_only":
            mode_instruksi = f"""
PENTING: Lembar ini HANYA memiliki Pilihan Ganda.
- Ekstrak TEPAT {jumlah_pg} jawaban pilihan ganda (A/B/C/D/E)
- jawaban_isian dan jawaban_essay kembalikan sebagai array kosong []
"""
        elif exam_mode == "pg_isian":
            mode_instruksi = f"""
PENTING: Lembar ini memiliki Pilihan Ganda DAN Isian Singkat.
- Ekstrak TEPAT {jumlah_pg} jawaban pilihan ganda (A/B/C/D/E)
- Ekstrak TEPAT {jumlah_isian} jawaban isian singkat
- jawaban_essay kembalikan sebagai array kosong []
"""
        else:  # full
            mode_instruksi = f"""
PENTING: Lembar ini memiliki Pilihan Ganda, Isian Singkat, DAN Essay.
- Ekstrak TEPAT {jumlah_pg} jawaban pilihan ganda (A/B/C/D/E)
- Ekstrak TEPAT {jumlah_isian} jawaban isian singkat
- Ekstrak TEPAT {jumlah_essay} jawaban essay
"""
        
        # Prompt utama yang diperkuat untuk tulisan cakar ayam
        prompt = f"""
Kamu adalah sistem OCR canggih yang SANGAT AHLI membaca tulisan tangan siswa Indonesia,
termasuk tulisan yang:
- Tidak rapi atau cakar ayam
- Huruf miring, tidak konsisten ukurannya  
- Ada coretan atau hapusan
- Tulisan terlalu kecil atau terlalu besar
- Huruf mirip satu sama lain (b/d, p/q, n/u, m/w, 1/i/l, 0/O)
- Tinta tipis atau memudar

{mode_instruksi}

STRATEGI MEMBACA TULISAN TIDAK JELAS:
1. Perhatikan KONTEKS - jawaban isian biasanya berupa kata/kalimat yang masuk akal dalam Bahasa Indonesia
2. Untuk huruf ambigu, pilih yang paling LOGIS secara kontekstual dan gramatikal
3. Jika ada coretan, baca tulisan yang PALING TERAKHIR ditulis
4. Untuk pilihan ganda, deteksi lingkaran/silang/contreng/titik di dekat huruf A/B/C/D/E
5. Jika pilihan ganda tidak jelas, tentukan dari POSISI tanda relatif terhadap huruf-huruf opsi
6. Jika benar-benar tidak terbaca, tulis "[tidak terbaca]"
7. JANGAN kosongkan field - selalu berikan interpretasi terbaik

CARA DETEKSI IDENTITAS:
- "Nama" atau "Nama Siswa" atau "Name" → isi di field "nama"
- "Kelas" atau "Class" → isi di field "kelas"  
- "Mapel" atau "Mata Pelajaran" atau "Pelajaran" atau "Bidang Studi" → isi di field "mapel"
- "Tanggal" atau "Hari/Tanggal" → isi di field "tanggal"

Format output WAJIB berupa JSON valid:
{{
  "nama": "nama lengkap siswa",
  "kelas": "kelas siswa",
  "mapel": "mata pelajaran",
  "tanggal": "tanggal ujian jika ada",
  "jawaban_pg": ["A","B","C","D",...],
  "jawaban_isian": ["jawaban1","jawaban2",...],
  "jawaban_essay": ["teks essay 1","teks essay 2",...],
  "catatan_keterbacaan": "bagian yang sulit dibaca dan cara interpretasinya"
}}

ATURAN WAJIB:
- Field "nama" WAJIB diisi dengan interpretasi terbaik, JANGAN pernah isi dengan "Siswa 1"
- Untuk pilihan ganda: kembalikan HANYA 1 huruf (A/B/C/D/E) per soal
- Untuk isian/essay: tulis hasil baca apa adanya termasuk kemungkinan typo siswa
- Kembalikan HANYA JSON valid, TANPA penjelasan tambahan, TANPA markdown
"""
        
        # Panggil AI Vision dengan detail tinggi
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=4096
        )
        
        result_text = response.choices[0].message.content.strip()
        result_data = clean_json_response(result_text)
        
        if not result_data:
            # Coba sekali lagi dengan prompt lebih sederhana
            retry_response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "high"
                                }
                            },
                            {
                                "type": "text",
                                "text": 'Baca lembar jawaban ini. Kembalikan JSON: {"nama":"...","kelas":"...","mapel":"...","tanggal":"...","jawaban_pg":[],"jawaban_isian":[],"jawaban_essay":[],"catatan_keterbacaan":"..."}'
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=2048
            )
            result_data = clean_json_response(retry_response.choices[0].message.content)
        
        if not result_data:
            result_data = {
                "nama": "[Tidak terdeteksi]",
                "kelas": "",
                "mapel": "",
                "tanggal": "",
                "jawaban_pg": [],
                "jawaban_isian": [],
                "jawaban_essay": [],
                "catatan_keterbacaan": "Gagal memproses gambar. Coba dengan foto yang lebih jelas."
            }
        
        result_data = validate_extraction(result_data)
        return jsonify(result_data)
    
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# =============================================
# ENDPOINT 2: Koreksi Teks Satu Soal
# =============================================
@app.route('/api/grade-text', methods=['POST'])
@rate_limit(max_requests=30, window=60)
def grade_text():
    try:
        data = request.get_json()
        
        jawaban_siswa = data.get('jawaban_siswa', '')
        kunci_jawaban = data.get('kunci_jawaban', '')
        tipe_soal = data.get('tipe_soal', 'isian')  # isian / essay
        bobot = data.get('bobot', 10)
        nomor_soal = data.get('nomor_soal', 1)
        
        if tipe_soal == 'isian':
            prompt = f"""
Kamu adalah guru yang mengoreksi jawaban siswa untuk soal isian singkat.
Pertimbangkan bahwa jawaban siswa mungkin memiliki typo atau ejaan tidak sempurna karena tulisan tangan.

Nomor Soal: {nomor_soal}
Kunci Jawaban: {kunci_jawaban}
Jawaban Siswa: {jawaban_siswa}
Bobot Nilai: {bobot}

Kriteria penilaian:
- Nilai PENUH ({bobot}): Jawaban benar atau sangat mendekati (typo kecil OK)
- Nilai SETENGAH ({bobot//2}): Jawaban sebagian benar atau konsep benar tapi tidak lengkap
- Nilai NOL (0): Jawaban salah atau kosong

Kembalikan JSON:
{{
  "nilai": <angka>,
  "feedback": "penjelasan singkat mengapa nilai tersebut",
  "jawaban_yang_benar": "{kunci_jawaban}"
}}
Kembalikan HANYA JSON, tanpa penjelasan tambahan.
"""
        else:  # essay
            prompt = f"""
Kamu adalah guru yang mengoreksi jawaban essay siswa.
Pertimbangkan bahwa jawaban siswa mungkin memiliki typo atau ejaan tidak sempurna.

Nomor Soal: {nomor_soal}
Kunci/Rubrik Jawaban: {kunci_jawaban}
Jawaban Siswa: {jawaban_siswa}
Bobot Nilai Maksimal: {bobot}

Kriteria penilaian essay:
- Nilai penuh: Semua poin utama tercakup, logis, tepat
- Nilai 75%: Sebagian besar poin tercakup
- Nilai 50%: Setengah poin tercakup atau ada pemahaman dasar
- Nilai 25%: Sedikit poin benar atau jawaban sangat singkat
- Nilai 0: Tidak relevan atau kosong

Kembalikan JSON:
{{
  "nilai": <angka>,
  "feedback": "penjelasan detail mengapa nilai tersebut dan apa yang kurang",
  "poin_terpenuhi": ["poin yang benar dari rubrik"],
  "poin_kurang": ["poin yang belum terpenuhi"]
}}
Kembalikan HANYA JSON, tanpa penjelasan tambahan.
"""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024
        )
        
        result = clean_json_response(response.choices[0].message.content)
        
        if not result:
            result = {
                "nilai": 0,
                "feedback": "Gagal mengevaluasi jawaban",
                "jawaban_yang_benar": kunci_jawaban
            }
        
        # Pastikan nilai tidak melebihi bobot
        if 'nilai' in result:
            result['nilai'] = min(float(result['nilai']), float(bobot))
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================
# ENDPOINT 3: Koreksi Batch (Multiple Soal)
# =============================================
@app.route('/api/grade-batch', methods=['POST'])
@rate_limit(max_requests=10, window=60)
def grade_batch():
    try:
        data = request.get_json()
        soal_list = data.get('soal_list', [])
        
        if not soal_list:
            return jsonify({"error": "soal_list diperlukan"}), 400
        
        # Batasi maksimal 20 soal per batch
        soal_list = soal_list[:20]
        
        # Buat prompt batch
        soal_text = ""
        for i, soal in enumerate(soal_list):
            soal_text += f"""
Soal {i+1}:
- Tipe: {soal.get('tipe', 'isian')}
- Kunci: {soal.get('kunci', '')}
- Jawaban Siswa: {soal.get('jawaban', '')}
- Bobot: {soal.get('bobot', 10)}
"""
        
        prompt = f"""
Kamu adalah guru yang mengoreksi jawaban siswa.
Pertimbangkan kemungkinan typo karena hasil baca tulisan tangan.

Koreksi semua soal berikut:
{soal_text}

Kembalikan JSON array:
[
  {{
    "nomor": 1,
    "nilai": <angka>,
    "feedback": "penjelasan singkat",
    "benar": true/false
  }},
  ...
]

Kriteria:
- Isian: Nilai penuh jika benar/mendekati, setengah jika sebagian, nol jika salah
- Essay: Nilai proporsional berdasarkan kelengkapan jawaban
- Pertimbangkan typo kecil sebagai benar untuk soal isian

Kembalikan HANYA JSON array, tanpa penjelasan tambahan.
"""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4096
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Bersihkan dan parse JSON array
        result_text = re.sub(r'```json\s*', '', result_text)
        result_text = re.sub(r'```\s*', '', result_text)
        result_text = result_text.strip()
        
        try:
            results = json.loads(result_text)
        except:
            # Cari array JSON
            array_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if array_match:
                results = json.loads(array_match.group())
            else:
                results = [{"nomor": i+1, "nilai": 0, "feedback": "Gagal mengevaluasi", "benar": False} 
                          for i in range(len(soal_list))]
        
        return jsonify({"results": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================
# ENDPOINT 4: Health Check
# =============================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "message": "SmartGrade AI Backend berjalan",
        "version": "2.0.0"
    })


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "app": "SmartGrade AI Backend",
        "endpoints": [
            "/api/extract-answers",
            "/api/grade-text", 
            "/api/grade-batch",
            "/health"
        ]
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
