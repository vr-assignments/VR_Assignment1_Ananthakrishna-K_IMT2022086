import numpy as np
import cv2
import imutils
import os

SOURCE_DIR = 'input'
RESULT_DIR = 'output'

def match_keypoints(kps1, kps2, desc1, desc2, match_ratio=0.8, reproj_thresh=5.0):
    brute_force_matcher = cv2.BFMatcher()
    preliminary_matches = brute_force_matcher.knnMatch(desc1, desc2, 2)
    
    valid_matches = []
    for pair in preliminary_matches:
        if len(pair) == 2 and pair[0].distance < pair[1].distance * match_ratio:
            valid_matches.append((pair[0].trainIdx, pair[0].queryIdx))
    
    if len(valid_matches) > 4:
        src_pts = np.float32([kps1[i] for (_, i) in valid_matches])
        dst_pts = np.float32([kps2[i] for (i, _) in valid_matches])
        
        homography, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
        return (valid_matches, homography, mask)
    return None

def draw_matches(img1, img2, kps1, kps2, matches, mask):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2
    
    for (train_idx, query_idx), valid in zip(matches, mask):
        if valid:
            pt1 = (int(kps1[query_idx][0]), int(kps1[query_idx][1]))
            pt2 = (int(kps2[train_idx][0]) + w1, int(kps2[train_idx][1]))
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 1)
    return canvas

def stitch_images(image_pair, match_ratio=0.5, reproj_thresh=5.0, display_matches=False):
    base_image, overlay_image = image_pair
    
    feature_extractor = cv2.SIFT_create()
    kps_base, desc_base = feature_extractor.detectAndCompute(base_image, None)
    kps_base = np.float32([kp.pt for kp in kps_base])
    
    kps_overlay, desc_overlay = feature_extractor.detectAndCompute(overlay_image, None)
    kps_overlay = np.float32([kp.pt for kp in kps_overlay])
    
    matched_data = match_keypoints(kps_overlay, kps_base, desc_overlay, desc_base, 
                                 match_ratio, reproj_thresh)
    if matched_data is None:
        print("Insufficient matching points detected")
        return None
    
    matches, homography, validity_mask = matched_data
    warped_image = cv2.warpPerspective(overlay_image, homography,
                                      (overlay_image.shape[1] + base_image.shape[1],
                                       base_image.shape[0]))
    warped_image[0:base_image.shape[0], 0:base_image.shape[1]] = base_image
    
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        warped_image = warped_image[y:y+h, x:x+w]
    
    if display_matches:
        match_visual = draw_matches(overlay_image, base_image, 
                                   kps_overlay, kps_base, matches, validity_mask)
        return (warped_image, match_visual)
    return warped_image

def create_panorama(input_path, output_path):
    image_files = [os.path.join(input_path, f) for f in os.listdir(input_path)]
    image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    if not image_files:
        print(f"No images found in {input_path}")
        return
    
    os.makedirs(output_path, exist_ok=True)
    
    base_img = cv2.imread(image_files[0])
    base_img = imutils.resize(base_img, width=200)
    
    for idx in range(1, len(image_files)):
        next_img = cv2.imread(image_files[idx])
        next_img = imutils.resize(next_img, width=200)
        
        result = stitch_images([base_img, next_img], display_matches=True)
        if result:
            base_img, match_vis = result
            cv2.imwrite(os.path.join(output_path, f"match_vis_{idx}.jpg"), match_vis)
    
    cv2.imwrite(os.path.join(output_path, "final_panorama.jpg"), base_img)

# Process all subdirectories in source folder
for subdir in os.listdir(SOURCE_DIR):
    dir_path = os.path.join(SOURCE_DIR, subdir)
    if os.path.isdir(dir_path):
        result_dir = os.path.join(RESULT_DIR, subdir)
        create_panorama(dir_path, result_dir)