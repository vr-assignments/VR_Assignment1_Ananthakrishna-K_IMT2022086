import cv2
import os
import numpy as np

SOURCE_DIR = 'input'
RESULT_DIR = 'output'

def process_directory(input_folder, result_folder):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        
    for image_name in os.listdir(input_folder):
        input_file = os.path.join(input_folder, image_name)
        output_file = os.path.join(result_folder, f"{os.path.splitext(image_name)[0]}_processed.jpg")
        
        processed_img, binary_img, scale_ratio = prepare_image(input_file)
        coin_candidates = extract_circular_shapes(binary_img, scale_ratio)
        final_image = extract_coins_from_background(processed_img, binary_img, coin_candidates)
        
        marked_image = final_image.copy()
        cv2.drawContours(marked_image, coin_candidates, -1, (0, 0, 255), 2)
        
        cv2.imwrite(output_file, marked_image)
        print(f"Image {image_name}: Detected {len(coin_candidates)} coins")

def prepare_image(image_path):
    source_img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    
    scale_ratio = 700 / max(source_img.shape[0], source_img.shape[1])
    
    resized_color = cv2.resize(source_img, None, fx=scale_ratio, fy=scale_ratio)
    resized_gray = cv2.resize(gray_img, None, fx=scale_ratio, fy=scale_ratio)
    
    # Apply blur and thresholding
    blurred = cv2.GaussianBlur(resized_gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    return resized_color, thresh, scale_ratio

def extract_circular_shapes(binary_image, scale_ratio):
    found_contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    coins = []
    min_size = 500 * (scale_ratio ** 2)
    
    for cnt in found_contours:
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        
        if perimeter > 0:
            # Filter by circularity and size
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if 0.7 < circularity < 1.2 and area > min_size:
                coins.append(cnt)
    
    return coins

def extract_coins_from_background(color_image, binary_image, coin_contours):
    coin_mask = np.zeros_like(binary_image)
    cv2.drawContours(coin_mask, coin_contours, -1, 255, cv2.FILLED)
    
    coin_pixels = cv2.bitwise_and(color_image, color_image, mask=coin_mask)
    
    output = np.zeros_like(color_image)
    output[coin_mask == 255] = coin_pixels[coin_mask == 255]
    
    return output

# Start processing
process_directory(SOURCE_DIR,RESULT_DIR)