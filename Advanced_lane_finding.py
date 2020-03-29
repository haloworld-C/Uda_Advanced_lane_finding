import numpy as np 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import cv2 
from helper_p2 import Camera_cali as cam
import helper_p2 as help
import pickle
# vedio related 
from moviepy.editor import VideoFileClip
# from IPython.display import HTML
'''
This profile aim to implement the pipeline Advanced_lane_finding.
All test case is here(You should test step by step).
Author: Jack halo
email: guangaltman@163.com
'''
# let's hit the road!
def get_lines_in_image(image):
    # declare the global variant
    global line
    # first receive the parameter from the preview image
    # TO DO:   
    # dist_src = "my_output/wide_dist_pickle.p"
    # read in the saved camera matrix and distortion coefficients
    # dist_pickle = pickle.load(open(dist_src, "rb"))
    # mtx = dist_pickle["mtx"] # camera matrix
    mtx = line.mtx
    # dist = dist_pickle["dist"] # distortion coefficients
    dist = line.dist
    
    # read in img
    # image = mpimg.imread(img_dir)
    
    # get the image demenssion
    ysize = image.shape[0]
    xsize = image.shape[1]
    
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply the undistort
    undistort = cam.undistort(image, mtx, dist)
    gray = cv2.cvtColor(undistort, cv2.COLOR_RGB2GRAY)
    # apply region mask, I define a vertices of ladder-shaped
    # this will work pefectly when the car is in the lane wihtout failure
    left_bottom = (50, ysize)
    right_bottom = (xsize-50, ysize)
    left_apex = (xsize//2 - 100, ysize//2 + 100)
    right_apex = (xsize//2 + 100, ysize//2 + 100)
    ytop = ysize//2 - 100
    vertices = np.array([[left_bottom, right_bottom, right_apex, left_apex]], dtype = np.int32) ##why [[ ]]
    region_mask = help.region_of_interest(undistort, vertices)
    # apply rgb space mask
    rgb_thresh = [(0, 200), (0,200), (0,50)]
    rgb_mask = help.rgb_select(region_mask, rgb_thresh)

    # apply HlS space
    hls_thresh = (100, 255)
    hls_mask = help.hls_select(region_mask, hls_thresh)
    # generate a binary image combined the rgb_mask and hls_mask
    color_combined = np.zeros_like(hls_mask)
    color_combined[(rgb_mask == 1) | (hls_mask == 1)] = 1

    # apply sobel filter
    # sobel filter -x
    sobel_thresh = (50, 300)
    sobel_filter_x = help.abs_sobel_thresh(gray,'x', 15, sobel_thresh)
    # sorber_filter_y = abs_sobel_thresh(gray,'y', 15, sober_thresh)  
    # sobel filter -magnitude
    magnitude_thresh = (30, 150)
    mag_filter = help.mag_thresh(gray, 9, magnitude_thresh)
    # sobel filter -arctan
    arctan_thresh = (0.7, np.pi/2)
    sobel_filter_tan = help.dir_threshold(mag_filter, 15, arctan_thresh)

    # apply the final combined binary image
    combined = np.zeros_like(color_combined)
    # combined : the color_combined add  sobel filter combined(filter the y direction
    # by sobel-x filter), TO DO: may be improved later.
    combined[(color_combined == 1) | \
        (((mag_filter==1) | (sobel_filter_tan ==1)) & (sobel_filter_x ==1))] = 1

    # apply region mask to final combined
    region_mask_combined = help.region_of_interest(combined, vertices)
    # unwarp to the top-view
    binary_warped, M, M_reverse = help.corners_unwarp(region_mask_combined)
    # find lines in top-view
    # out_img, text = help.fit_poly(binary_warped, ytop, line)
    out_img, text1, text2, len_points = help.search_around_poly(binary_warped, line)
    # warp back in real image to display
    re_out_img = help.conners_warp(out_img, M_reverse)
    # filter the disturbance introduced by warp with region_mask
    re_out_img = help.region_of_interest(re_out_img, vertices)
    cv2.putText(re_out_img, text1, (50, 50), cv2.FONT_HERSHEY_PLAIN,\
         2.0, (255, 0, 0), 2, bottomLeftOrigin=False)
    cv2.putText(re_out_img, text2, (50, 100), cv2.FONT_HERSHEY_PLAIN,\
         2.0, (255, 0, 0), 2, bottomLeftOrigin=False)     
    cv2.putText(re_out_img,str(len(len_points)),(1000,50),cv2.FONT_HERSHEY_PLAIN,\
         5.0, (255, 0, 0), 2, bottomLeftOrigin=False)
    # final result
    add_weight = help.weighted_img(re_out_img, undistort)
    mask_image = np.dstack((region_mask_combined, region_mask_combined, region_mask_combined))*255
    pinjie = np.concatenate((add_weight, mask_image), axis = 1)
    return pinjie
#  def process_image(image):
#     result = get_lines_in_image(img_dir, mtx, dist)
#     return result   

def get_lines_in_vedio(dst, out_dir):
    # clip1 = VideoFileClip(dst).subclip(0,5)
    clip1 = VideoFileClip(dst)
    white_clip = clip1.fl_image(get_lines_in_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(out_dir, audio=False)


## Always test!
straight1 = "./test_images/straight_lines1.jpg"
straight2 = "./test_images/straight_lines2.jpg"
test1 = "./test_images/test1.jpg"
test2 = "./test_images/test2.jpg"
test3 = "./test_images/test3.jpg"
test4 = "./test_images/test4.jpg"
test5 = "./test_images/test5.jpg"
test6 = "./test_images/test6.jpg"

dist_src = "my_output/wide_dist_pickle.p"
# read in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load(open(dist_src, "rb"))
mtx = dist_pickle["mtx"] # camera matrix
dist = dist_pickle["dist"] # distortion coefficients
# define a line() class to store parameters
line = help.line(mtx, dist)
# result = get_lines_in_image(test3, dist_src, mtx, dist)
# dst = "project_video.mp4"
dst = "challenge_video.mp4"
# dst = "harder_challenge_video.mp4"
out_dir = "./my_output/project_out1.mp4"
get_lines_in_vedio(dst, out_dir)

'''
# img = mpimg.imread(straight1)
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 9))
f.tight_layout()
ax1.imshow(undistort)
ax1.set_title('original image')
ax2.imshow(result)
ax2.set_title('undistorted image')
plt.subplots_adjust(left=0., right=1, top=1, bottom=0.)

# plt.imshow(undistort)
# plt.show()
'''