## P4 - Advanced lane lines detection

### This project uses advanced computer vision techniques to detect and draw lane lines in a video. 

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[image1]: ./output_images/download.png "Undistorted"
[image2]: ./output_images/download(1).png "Road Transformed"
[image3]: ./output_images/download(2).png "Binary Example"
[image4]: ./output_images/download(3).png "Warp Example"
[image5]: ./output_images/download(5).png  "Fit Visual"
[image6]: ./output_images/download(6).png  "Output"
[image7]: ./output_images/download(7).png  "Output"
[image8]: ./output_images/download(9).png  "Output"
[image10]: ./output_images/download(10).png  "Output"
[image11]: ./output_images/download(11).png  "Output"
[video1]: ./results/video_out_4.mp4 "Video"

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view)

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### README

## PIPELINE: Calibrate > Undistort > Color and Gradient threshold > Perspective transform

### Camera Calibration

The code for this step is contained in the helper_ module. The camera_calibration( ) function loads in all of the  images provided, finds the  corners, `objpoints`, and then computes the calibration matrix by calling `cv2.calibrateCamera()`. This can be then used with the `cv2.undistort()` function to  every image in our pipeline. 

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Distortion-corrected image.

![alt text][image2]

#### 2. Color transforms, gradients and other methods to create a thresholded binary image. 

![alt text][image11]

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 34 through 44 in `main.py`).  Here's an example of my output for this step.

![alt text][image3]

Here is the code and thresholds I used in my pipeline: 

```python 
	ksize = 5
    dir_binary = dir_threshold(img_bgr, sobel_kernel=ksize, thresh=(0.7, 1.3))
    sobelx_binary = mag_thresh(img_bgr, sobel_kernel=3, mag_thresh=(20, 100))
    s_binary = s_channel_threshold(img_bgr, s_thresh=(170, 255))

    combined = np.zeros_like(dir_binary)
    combined[((sobelx_binary == 1) & (dir_binary == 1))] = 1

    combined_binary = np.zeros_like(combined)
    combined_binary[(s_binary == 1) | (combined == 1)] = 1
```

#### 3. Perspective transform

The transform is being done in lines 48 through 51 of the `main.py` function. `cv2.getPerspectiveTransform()` takes (`src`) and destination (`dst`) points as arguments to compute the transformation matrix M, which is then used with the `cv2.warpPerspective()` function to preform the perspective transform.  I chose the hardcode the source and destination points in the following manner:

|   Source    | Destination |
| :---------: | :---------: |
| [589, 455]  |  [290, 0]   |
| [695, 455]  |  [1020, 0]  |
| [1020, 663] | [1020, 720] |
| [290, 663]  | [290, 720]  |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Identify lane-line pixels and fit their positions with a polynomial.

- Do the perspective transform on the thresholded image. 
- Take the bottom half and get a histogram of all columns as shown below.
- The two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines.

![alt text][image6]
- Split the image into x sections and start the search from the x positins found above
- Define a margin for the search window and find the number of nonzero pixels in the window
- If the number of pixels > n move to another section and recenter the window to the mean position of the previous window:
   ![alt text][image7]

- Split the image into two halfs and fit a line to the hot pixels found.
   ![alt text][image5]
- Once you have a fit, just search around the curves found.

#### 5. Calculate the radius of curvature of the lane and the position of the vehicle with respect to center.

I use this conversion to translate the pixels dimensions to meters, it's based on the width of a standart US lane and a length of dashed line. 

```python
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
```
 ![alt text][image8]


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

To find the line curvature, the following formula cand be used:
![alt text][image8]

A reasonable comparison can be found from the map of the road where the video was shot:

![alt text][image10]

---

### Pipeline (video)

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Problems / issues and suggestions for improvement.

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
