import cv2
import numpy as np
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

# Initialize OCR
ocr = PaddleOCR(
    lang='en',
    use_textline_orientation=True
)

pdf_path = r"D:\chandra_ocr\project_env\ilovepdf_split\student-9.pdf"

images = convert_from_path(pdf_path, dpi=300)

full_transcript = []

print(f"Processing {len(images)} pages...")

for i, page in enumerate(images):
    print(f"Processing Page {i+1}...")
    
    img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

    # Optional Preprocessing (Boost accuracy)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    result = ocr.ocr(thresh)

    page_text = ""
    if result[0] is not None:
        for line in result[0]:
            page_text += line[1][0] + "\n"

    full_transcript.append(f"\n--- Page {i+1} ---\n{page_text}")

print("\n" + "="*20 + " FINAL OUTPUT " + "="*20)
print("\n".join(full_transcript))