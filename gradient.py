import argparse
import cv2
import imutils
import os
import numpy as np
import pytesseract

def detect_plate_text(image):
    """辨識並回傳車牌文字"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blurred, 30, 200)
    contours = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    plate_text = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 2.0 <= aspect_ratio <= 5.5:
                plate_region = gray[y:y+h, x:x+w]
                _, plate_binary = cv2.threshold(plate_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                plate_text = pytesseract.image_to_string(plate_binary, config='--psm 8 --oem 3').strip()
                plate_text = ''.join(c for c in plate_text if c.isalnum())
                if plate_text:
                    return plate_text
    return None

def text_to_image(text, width=300, height=100, font_scale=1, thickness=2):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(canvas, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    return canvas

def apply_morphological_operations(image):
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(image, kernel, iterations=1)
    erosion = cv2.erode(image, kernel, iterations=1)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return dilation, erosion, opening, closing

def detect_contours(image, edge_map):
    contours = cv2.findContours(edge_map.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    contour_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    return contour_img, len(contours)

def apply_thresholding(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, adaptive, otsu

def create_combined_display(images_dict, max_width=2000):
    processed_images = {}
    target_height = 600
    
    for name, img in images_dict.items():
        if img is None:
            continue
        if isinstance(img, str):  # 如果是字串，跳過
            continue
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        aspect_ratio = img.shape[1] / img.shape[0]
        target_width = int(target_height * aspect_ratio)
        resized = cv2.resize(img, (target_width, target_height))
        processed_images[name] = resized
    
    n_images = len(processed_images)
    n_cols = min(3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    cell_width = max(img.shape[1] for img in processed_images.values())
    cell_height = max(img.shape[0] for img in processed_images.values())
    canvas = np.zeros((cell_height * n_rows, cell_width * n_cols, 3), dtype=np.uint8)
    
    for idx, (name, img) in enumerate(processed_images.items()):
        i, j = idx // n_cols, idx % n_cols
        y, x = i * cell_height, j * cell_width
        img_with_text = img.copy()
        cv2.putText(img_with_text, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 3)
        canvas[y:y+img.shape[0], x:x+img.shape[1]] = img_with_text
    
    return canvas

# 主程式
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, nargs='+')
args = vars(ap.parse_args())

for image_path in args["images"]:
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found!")
        continue

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    wide = cv2.Canny(blurred, 10, 200)
    tight = cv2.Canny(blurred, 350, 400)
    auto = imutils.auto_canny(blurred)
    dilation, erosion, opening, closing = apply_morphological_operations(gray)
    binary, adaptive, otsu = apply_thresholding(gray)
    contour_img, num_contours = detect_contours(image, auto)
    
    plate_text = detect_plate_text(image)
    if plate_text:
        print(f"偵測到的車牌號碼: {plate_text}")
        plate_image = text_to_image(plate_text)
    else:
        print("未能偵測到車牌")
        plate_image = text_to_image("No Plate Detected")

    images_dict = {
        "Original": image,
        "Plate Numbers": plate_image,
        f"Contours ({num_contours})": contour_img,
        "Adaptive": adaptive,
        "Otsu": otsu,
        "Dilation": dilation,
        "Erosion": erosion,
        "Opening": opening,
        "Closing": closing,
        "Wide Edge": wide,
        "Tight Edge": tight,
    }
    
    combined_display = create_combined_display(images_dict)
    cv2.imshow(f"Image Processing Results - {image_path}", combined_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

