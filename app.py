from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
from omr_engine import process_answer_sheet

app = Flask(__name__)
CORS(app)  # Izinkan request dari Blogger (cross-origin)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "SmartGrade OMR API is running"})

@app.route("/api/scan", methods=["POST"])
def scan_answer_sheet():
    try:
        data = request.get_json()

        if "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 image dari frontend
        image_data = data["image"].split(",")[1]  # Hapus prefix "data:image/..."
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        total_questions = data.get("total_questions", 20)
        num_choices = data.get("num_choices", 5)  # A-E = 5 pilihan

        # Proses OMR
        answers = process_answer_sheet(img, total_questions, num_choices)

        return jsonify({
            "success": True,
            "answers": answers,          # Contoh: "ABCDABCDABCDABCDABCD"
            "total_detected": len(answers)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
