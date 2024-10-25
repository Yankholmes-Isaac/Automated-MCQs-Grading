import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
app = Flask (__name__)
CORS(app)

#Global variable to store the answer key
stored_answer_key = []

def find_largest_rectangle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            return approx

    return None


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped, M


def preprocess_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh


def detect_bubbles(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 1000:  # Adjust these values based on your image size
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 0.8 < aspect_ratio < 1.2:  # Assuming bubbles are roughly circular
                bubbles.append((x, y, w, h))

    return bubbles


def organize_bubbles(bubbles, num_questions):
    # Sort bubbles by y-coordinate (row) first, then x-coordinate (column)
    sorted_bubbles = sorted(bubbles, key=lambda b: (b[1], b[0]))

    # Split into two columns
    mid_x = max(b[0] for b in sorted_bubbles) // 2
    left_bubbles = [b for b in sorted_bubbles if b[0] < mid_x]
    right_bubbles = [b for b in sorted_bubbles if b[0] >= mid_x]

    # Organize bubbles into questions
    organized = []
    for column in [left_bubbles, right_bubbles]:
        for i in range(0, len(column), 4):
            question_bubbles = sorted(column[i:i + 4], key=lambda b: b[0])
            organized.append(question_bubbles)

    return organized[:num_questions]

def grade_exam(thresh, organized_bubbles, answer_key):
    student_answers = []
    for question_bubbles in organized_bubbles:
        chosen_answer = -1
        max_filled = 0
        for i, (x, y, w, h) in enumerate(question_bubbles):
            roi = thresh[y:y + h, x:x + w]
            filled_ratio = np.sum(roi == 255) / roi.size
            if filled_ratio > max_filled:
                max_filled = filled_ratio
                chosen_answer = i
        student_answers.append(chosen_answer)

    score = sum(1 for sa, ca in zip(student_answers, answer_key) if sa == ca)
    return student_answers, score


def annotate_image(image, organized_bubbles, student_answers, answer_key):
    for i, (question_bubbles, student_answer, correct_answer) in enumerate(
            zip(organized_bubbles, student_answers, answer_key)):
        for j, (x, y, w, h) in enumerate(question_bubbles):
            center = (x + w // 2, y + h // 2)
            if j == student_answer:
                color = (0, 255, 0) if student_answer == correct_answer else (0, 0, 255)
                cv2.circle(image, center, w // 2, color, 2)
            if j == correct_answer and student_answer != correct_answer:
                cv2.circle(image, center, w // 2, (255, 0, 0), 2)

    return image

def add_score_to_image(image, score_view, total_questions):
    h, w = image.shape[:2]
    score_text = f"Score: {score_view}/{total_questions}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(score_text, font, font_scale, font_thickness)[0]

    # Position the text in the top right corner with some padding
    text_x = w - text_size[0] - 10
    text_y = text_size[1] + 10

    cv2.putText(image, score_text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)
    return image


@app.route('/upload_answer_key', methods=['POST'])
def upload_answer_key():
    global stored_answer_key
    data = request.json
    answer_key = data.get('answer_key', [])
    if not answer_key:
        return jsonify({"success": False, "message": "No answer key provided"}), 400

    # Convert answer key to indices (A -> 0, B -> 1, etc.)
    stored_answer_key = [ord(answer.upper()) - ord('A') for answer in answer_key]
    return jsonify({"success": True, "message": "Answer key uploaded successfully"}), 200

@app.route('/grade_exam', methods=['POST'])
def grade_exam_api():
    global stored_answer_key

    # Check if answer key is available
    if not stored_answer_key:
        return jsonify({"success": False, "message": "Answer key not uploaded"}), 400

    # Get the image from the request
    image_data = request.json.get("image")
    if not image_data:
        return jsonify({"success": False, "message": "No image data provided"}), 400

    # Decode the image
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the image
    rect = find_largest_rectangle(image)
    if rect is None:
        return jsonify({"success": False,"message": "Could not find the answer sheet in the image"}), 400


    roi, M = four_point_transform(image, rect.reshape(4, 2))
    thresh = preprocess_roi(roi)
    bubbles = detect_bubbles(thresh)
    organized_bubbles = organize_bubbles(bubbles, len(stored_answer_key))

    # Grade the exam
    student_answers, score = grade_exam(thresh, organized_bubbles, stored_answer_key)

    # Annotated the image
    annotated_roi = annotate_image(roi.copy(), organized_bubbles, student_answers, stored_answer_key)
    h, w = image.shape[:2]
    final_annotated = cv2.warpPerspective(annotated_roi, np.linalg.inv(M), (w,h))
    mask = np.all(final_annotated == [0, 0, 0], axis =-1)
    final_result = np.where(mask[..., None], image, final_annotated)
    final_result = add_score_to_image(final_result, score, len(stored_answer_key))

    # Encode the result image to base64
    _, buffer = cv2.imencode('.jpg', final_result)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "success": True,
        "message": "Exam graded successfully",
        "score": score,
        "total_questions": len(stored_answer_key),
        "graded_image": jpg_as_text
    }), 200

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, debug=True)







