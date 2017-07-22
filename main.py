import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

from helper_functions import camera_calibration, plot_before_after, mag_thresh, dir_threshold, s_channel_threshold
from line_detection import find_lanes, Line, update_lanes
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# PIPELINE: Calibrate > Undistort > Color and Gradient threshold > Perspective transform

# Get data for calibration
images = glob.glob('camera_cal/calibration*.jpg')
mtx, dist = camera_calibration(images, nx=9, ny=6)


img_bgr = cv2.imread('test_images/straight_lines1.jpg')
#img_bgr = cv2.undistort(img_bgr, mtx, dist, None, mtx)
#cv2.imwrite('test_images/undistorted_straight.jpg', img_bgr)

#undist = cv2.undistort(img_bgr, mtx, dist, None, mtx)
Right = Line()
Left = Line()
# Color and Gradient threshold
#img_bgr = cv2.imread('test_images/test3.jpg')
def process_image(img_bgr):
    img_bgr = cv2.undistort(img_bgr, mtx, dist, None, mtx)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    ksize = 5
    dir_binary = dir_threshold(img_bgr, sobel_kernel=ksize, thresh=(0.7, 1.3))
    sobelx_binary = mag_thresh(img_bgr, sobel_kernel=3, mag_thresh=(20, 100))
    s_binary = s_channel_threshold(img_bgr, s_thresh=(170, 255))


    combined = np.zeros_like(dir_binary)
    combined[((sobelx_binary == 1) & (dir_binary == 1))] = 1

    combined_binary = np.zeros_like(combined)
    combined_binary[(s_binary == 1) | (combined == 1)] = 1

    img_size = (img_bgr.shape[1], img_bgr.shape[0])

    src = np.float32([[589, 455], [695, 455], [1020, 663], [290, 663]])
    dst = np.float32([[290, 0], [1020, 0], [1020, 720], [290, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    binary_warped = cv2.warpPerspective(combined_binary, M, img_size, flags=cv2.INTER_LINEAR)
    last_left_fit = Left.current_fit
    last_right_fit = Right.current_fit
    #plot_before_after(combined_binary, binary_warped)
    if Right.detected == 1 and Left.detected == 1:
        Left.current_fit, Right.current_fit, Left.radius_of_curvature, Right.radius_of_curvature = update_lanes(Left.current_fit, Right.current_fit, binary_warped)
    else: Left.current_fit, Right.current_fit, Left.radius_of_curvature, Right.radius_of_curvature = find_lanes(binary_warped)
    xm_per_pix = 3.7 / 700
    top_width = (Right.current_fit[2] - Left.current_fit[2]) * xm_per_pix
    bottom_width = ((Right.current_fit[0] * 720 ** 2 + Right.current_fit[1] * 720 + Right.current_fit[2]) - \
                    (Left.current_fit[0] * 720 ** 2 + Left.current_fit[1] * 720 + Left.current_fit[2])) * xm_per_pix
    Left.parallel.append([top_width, bottom_width])


    if abs(top_width - bottom_width) > 0.5 and last_left_fit[0] != [False] and abs(Left.radius_of_curvature - Right.radius_of_curvature) < 500:
        Right.current_fit = last_right_fit
        Left.current_fit = last_left_fit

    if abs(Left.radius_of_curvature - Right.radius_of_curvature) < 500 and abs(top_width - bottom_width) < 0.5:
        Right.detected = 1
        Left.detected = 1
    else:
        Right.detected = 0
        Left.detected = 0

    Left.ave_radius.append(Left.radius_of_curvature)
    Right.ave_radius.append(Right.radius_of_curvature)

    if len(Right.ave_radius) > 20:
        ave_rad_right = sum(Right.ave_radius[-20:])/20
        ave_rad_left = sum(Left.ave_radius[-20:])/ 20
    else:
        ave_rad_right = sum(Right.ave_radius)/len(Left.ave_radius)
        ave_rad_left = sum(Left.ave_radius)/len(Left.ave_radius)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = Left.current_fit[0] * ploty ** 2 + Left.current_fit[1] * ploty + Left.current_fit[2]
    right_fitx = Right.current_fit[0] * ploty ** 2 + Right.current_fit[1] * ploty + Right.current_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img_bgr.shape[1], img_bgr.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img_bgr, 1, newwarp, 0.3, 0)

    Right.line_base_pos = ((Right.current_fit[0] * 720 ** 2 + Right.current_fit[1] * 720 + Right.current_fit[
        2])) * xm_per_pix
    Left.line_base_pos = ((Left.current_fit[0] * 720 ** 2 + Left.current_fit[1] * 720 + Left.current_fit[
        2])) * xm_per_pix

    off_center = abs((Right.line_base_pos - Left.line_base_pos) / 2 + Left.line_base_pos) - img_size[1] * xm_per_pix

    cv2.putText(result, 'Off center: {}'.format(off_center), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    cv2.putText(result, 'Average radius L/R: {} / {}'.format(round(ave_rad_left, -2), round(ave_rad_right, -2)), (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    return result

# result = process_image(img_bgr)
# plt.imshow(result)
# plt.show()

white_output = 'results/video_out_4.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!

white_clip.write_videofile(white_output, audio=False)

print(Left.parallel)
