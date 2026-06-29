import cv2
import numpy as np

def order_points(pts):
    """Urutkan 4 titik sudut: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Koreksi perspektif agar lembar jawab tegak lurus."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))

def process_answer_sheet(image, total_questions=20, num_choices=5):
    """
    Pipeline utama OMR:
    1. Preprocessing
    2. Deteksi kontur lembar
    3. Koreksi perspektif
    4. Ekstraksi jawaban dari bubble
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # Temukan kontur terbesar (area lembar jawab)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    sheet_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            sheet_contour = approx
            break

    if sheet_contour is None:
        # Jika tidak terdeteksi, proses gambar langsung tanpa warp
        warped = cv2.resize(gray, (600, 800))
    else:
        warped = four_point_transform(gray, sheet_contour.reshape(4, 2))
        warped = cv2.resize(warped, (600, 800))

    # Threshold untuk isolasi bubble
    _, thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Bagi area jawaban menjadi grid (questions x choices)
    answers = []
    row_height = thresh.shape[0] // total_questions
    col_width = thresh.shape[1] // num_choices

    for q in range(total_questions):
        bubble_counts = []
        for c in range(num_choices):
            # Crop area setiap bubble
            y1 = q * row_height + 5
            y2 = (q + 1) * row_height - 5
            x1 = c * col_width + 5
            x2 = (c + 1) * col_width - 5

            bubble = thresh[y1:y2, x1:x2]
            count = cv2.countNonZero(bubble)
            bubble_counts.append(count)

        # Pilihan dengan pixel terbanyak = jawaban yang diisi
        chosen = np.argmax(bubble_counts)
        max_fill = bubble_counts[chosen]
        total_area = row_height * col_width

        # Hanya dianggap diisi jika pixel > 15% area bubble
        if max_fill > total_area * 0.15:
            answers.append(chr(65 + chosen))  # 0->A, 1->B, dst
        else:
            answers.append("?")  # Tidak terbaca

    return "".join(answers)
