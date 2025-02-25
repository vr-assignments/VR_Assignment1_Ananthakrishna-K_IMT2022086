# VR Assignment: Coin Detection & Panorama Stitching

## Overview

This assignment consists of two parts:

1. Coin Detection and Counting – Detect, segment, and count coins in a given image.
2. Panorama Stitching – Stitch multiple images together to create a panorama.

## Part 1: Coin Detection and Counting

This part of the assignment processes images to detect and count coins using OpenCV.

### Implementation Steps:
1. Load images from the `input` directory.
2. Convert images to grayscale and resize them for processing.
3. Apply Gaussian blur and adaptive thresholding.
4. Detect circular shapes using contour approximation.
5. Extract coin regions and overlay detected contours on the output image.
6. Save processed images in the `output` directory.

### Dependencies:
- OpenCV
- NumPy
 ```
 pip install numpy opencv-python
 ```
### Running the Code:
To run the coin detection script:
``` 
cd part1
python script.py
```

- Ensure the `input` directory contains images of coins.
- The number of detected coins is printed in the console.
- Marked images with detected coins are saved in the `output/` directory.


## Part 2: Panorama Stitching
--------------------------
This part stitches a sequence of images together to form a panorama.

### Implementation Steps:
1. Load images from the `input` directory.
2. Extract keypoints and descriptors using SIFT.
3. Match keypoints between consecutive images.
4. Compute homography and warp images for alignment.
5. Crop the final stitched image to remove black borders.
6. Save the final panorama in the `output` directory.

### Dependencies:
- OpenCV
- NumPy
- Imutils
```
 pip install numpy opencv-python imutils
```

### Running the Code:

To run the panorama stitching script:

```
cd part2
python script.py
```

- Make sure the input directory is structured correctly. Each folder inside input/ should contain images to be stitched, named sequentially (e.g., `0.jpg`, `1.jpg`, `2.jpg`).
- The final panorama for each folder is saved in `output/<folder_name>/final_panorama.jpg`.
- The `output/<folder_name>/` directory also contains images showing matching keypoints between adjacent images.


## Notes:
------
- Ensure that the 'input' folder is correctly structured before running the scripts.
- The order of images in the panorama stitching process should follow a sequence for better results.
- More details about the implementation can be found in the report within the same repository.