import anthropic
from groq import Groq
import base64
import json
import re

def extract_answers_from_image(image_base64, exam_mode="full"):
    """
    Enhanced extraction dengan kemampuan membaca tulisan tidak jelas
    """
    
    # Prompt yang diperkuat untuk tulisan cakar ayam
    prompt = """
Kamu adalah sistem OCR canggih yang SANGAT AHLI membaca tulisan tangan siswa Indonesia,
termasuk tulisan yang:
- Tidak rapi atau cakar ayam
- Huruf miring, tidak konsisten ukurannya
- Ada coretan atau hapusan
- Tulisan terlalu kecil atau terlalu besar
- Huruf yang mirip satu sama lain (b/d, p/q, n/u, m/w, 1/i/l, 0/O)

TUGAS KAMU:
Baca lembar jawaban ini dan ekstrak semua informasi dengan strategi berikut:

STRATEGI MEMBACA TULISAN TIDAK JELAS:
1. Perhatikan KONTEKS - jawaban isian biasanya berupa kata/kalimat yang masuk akal
2. Untuk huruf yang ambigu, pilih yang paling LOGIS secara kontekstual
3. Jika ada coretan, baca tulisan yang PALING TERAKHIR ditulis
4. Untuk pilihan ganda, deteksi lingkaran/silang/tanda apapun di dekat huruf A/B/C/D/E
5. Jika benar-benar tidak terbaca sama sekali, tulis "[tidak terbaca]"
6. JANGAN kosongkan field - selalu berikan interpretasi terbaik

YANG HARUS DIEKSTRAK:
{
  "nama": "nama lengkap siswa - cari label Nama/Nama Siswa/Name, baca semua kemungkinan interpretasi tulisan",
  "kelas": "kelas siswa - cari label Kelas/Class",
  "mapel": "mata pelajaran - cari label Mapel/Mata Pelajaran/Pelajaran",
  "tanggal": "tanggal jika ada",
  "jawaban_pg": ["jawaban nomor 1","jawaban nomor 2",...],
  "jawaban_isian": ["jawaban isian 1","jawaban isian 2",...],
  "jawaban_essay": ["jawaban essay 1","jawaban essay 2",...],
  "catatan_keterbacaan": "catat bagian mana yang sulit dibaca dan interpretasinya"
}

PENTING:
- Untuk pilihan ganda: kembalikan HANYA huruf tunggal (A/B/C/D/E)
- Jika pilihan ganda tidak jelas, tebak dari posisi tanda yang paling dekat dengan huruf mana
- Untuk isian/essay: tulis hasil baca apa adanya, termasuk kemungkinan typo siswa
- Field "nama" WAJIB diisi, minimal tulis interpretasi terbaik tulisan di area nama
- JANGAN pernah mengembalikan "Siswa 1" sebagai nama

Kembalikan HANYA JSON valid, tanpa penjelasan tambahan.
"""

    try:
        client = Groq()
        
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",  # Model vision terbaru Groq
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"  # Gunakan detail tinggi untuk tulisan tidak jelas
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            temperature=0.1,  # Rendah = lebih konsisten dan tidak mengarang
            max_tokens=4096
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Bersihkan response dari markdown jika ada
        result_text = re.sub(r'```json\s*', '', result_text)
        result_text = re.sub(r'```\s*', '', result_text)
        result_text = result_text.strip()
        
        data = json.loads(result_text)
        
        # Validasi: pastikan nama tidak fallback ke default
        if not data.get('nama') or data['nama'] in ['', 'Siswa 1', 'Unknown', 'N/A']:
            data['nama'] = '[Nama tidak terbaca - harap isi manual]'
        
        return data
        
    except json.JSONDecodeError:
        # Coba ekstrak JSON dari response yang tidak bersih
        return extract_json_fallback(result_text)
    except Exception as e:
        return {"error": str(e)}


def extract_json_fallback(text):
    """
    Fallback parser jika JSON tidak valid
    """
    try:
        # Cari pola JSON dalam teks
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    return {
        "nama": "[Tidak terdeteksi]",
        "kelas": "[Tidak terdeteksi]",
        "mapel": "[Tidak terdeteksi]",
        "jawaban_pg": [],
        "jawaban_isian": [],
        "jawaban_essay": [],
        "catatan_keterbacaan": "Gagal memproses gambar"
    }
