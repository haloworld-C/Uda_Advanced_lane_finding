# difne a camera class to get and store the 
# camera related parameters
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
class Camera_cali():
    '''
    this class initral the camera-trans related parameter
    and caculate these para by functions below.
    '''
    def __init__(self, x, y):
        # x, y is the demension of points (width , length)        
        self.x = x
        self.y = y
        
        
    def caculate_para(self, dir_dst):
        '''
        @ self.mtx , the transtion matrax of camera
        @ self.dist, the dist_coeffs
        '''    
        # prepare the object points which standing for a standard chessboard
        objp = np.zeros((self.x*self.y, 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.x, 0:self.y].T.reshape(-1, 2)
        #define lists to store object points
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane
        # get the camera images as input
        images = glob.glob(dir_dst)
        # step through the list and search for chessboard conners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            img_size = (img.shape[1],img.shape[0])
            gray = cv2.cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # find the chessboard conners
            ret, conners = cv2.findChessboardCorners(gray, (self.x, self.y), None)

            if ret == True:
                objpoints.append(objp)
                imgpoints.append(conners)
        # caculate the para
        ret, self.mtx, self.dist, rvecs, tvecs = \
            cv2.calibrateCamera(objpoints, imgpoints,img_size, None, None)
          
    # undistort function
    def undistort(img, mtx, dist):
        '''
        this func undistort a  given image with the parameters caculated
        '''
        # image = cv2.imread(img)
        return cv2.undistort(img, mtx, dist, None, mtx)

    # save data in pickle file
    def save_as_pickle(self, out_dst):
        '''
        save the data in a pickle file in the route of 
        <out_dst>, this only need to caculate onece
        '''
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = self.mtx
        dist_pickle["dist"] = self.dist
        pickle.dump( dist_pickle, open( out_dst, "wb" ) )

class line():
    '''
    this class is define to store some useful parameter 
    and reuse in next image frame work
    '''
    def __init__(self, mtx, dist):
        self.confidence_r = 0
        self.confidence_l = 0
        self.left_fit = [0, 0, 0]
        self.right_fit = [0, 0, 0]
        self.best_fit_left = [0, 0, 0]
        self.best_fit_right = [0, 0, 0]
        self.recent_fit_left = []
        self.recent_fit_right = []
        self.last_radius = [1000,1000] # the real radius of this radio
        # self.base_width = 0 # the width between is a fixed number which can be
                                # taken advantages
        self.lane_margin = 100 # set the line region filter of each line
        self.mtx = mtx
        self.dist = dist
        self.first = True # judge whether is fist image
        self.wide_base = 0
        self.width = 0
    def get(self):
        return self.confidence, self.right_fit, self.left_fit, self.base_width,\
                self.last_radius, self.last_radius


# different process methods of image, 
# they all return a RGB-GRAY bininary output


# region mask
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask) # also a combined way!
    return masked_image

# color space 
# color space of RGB
def rgb_select(img, thresh = [255,255,255]):
    # receive a image in RGB mode
    # apply a threshold to defined channel
    # return a binary channel
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    # define a same demenssion of img to apply mask
    binary = np.zeros_like(img)
    binary[~((R < thresh[0][1]) | (G < thresh[1][1]) | (B < thresh[2][1]))] = 1
    binary = cv2.cvtColor(binary, cv2.COLOR_RGB2GRAY)
    return binary
# color space of HLS(specificly s channel)
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = HLS[:,:,2]
    # Binary_img = np.zeros_like(S)
    # Binary_img[(S > thresh[0]) and (S <= thresh[1])] = 1
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
     
    return binary
# sobel_mag
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # sobel_xy = sobel_x + sobel_y*cmath.sqrt(-1)
    # abs_sobel = np.absolute(sobel_xy)
    abs_sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    sxbinary = np.zeros_like(scaled_sobel)
    # here is a mash based on sobel_scale
    sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return sxbinary
# sobel_tan
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    arctan = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    # scaled_arctan = np.uint8(255*arctan/np.max())
    arctan_mask = np.zeros_like(arctan)
    arctan_mask[(arctan >= thresh[0]) & (arctan <= thresh[1])] = 1

    return arctan_mask
# sobel_x
def abs_sobel_thresh(img, orient='x', sobel_kernel = 3, thresh=(0,255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    else:
        raise 'wrong input: not x or y'
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    sxbinary = np.zeros_like(scaled_sobel)
    # here is a mash based on sobel_scale
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return sxbinary

# get the transform matrix
def corners_unwarp(img):
    '''
    @return warped
    @return M
    @param img
    @param nx
    @param ny
    @param mtx
    @param dist
    '''
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_size = (img.shape[1], img.shape[0])
    # dst = cv2.undistort(gray, mtx, dist, None, mtx)
    
    # ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
    #print(corners)
    # if ret == True:
        # cv2.drawChessboardCorners(img, (8,6), corners, ret)
    src = np.float32(
                    [[712, 464],
                    [936, 612],
                    [368, 615],
                    [572, 464]])
    dst_point = np.float32(
                    [[960, 70],
                    [960, 630],
                    [320, 630],
                    [320, 70]])
    # src = np.float32(
    #                 [[669, 437],
    #                 [1020, 665],
    #                 [289, 665],
    #                 [604,437]])
    # dst_point = np.float32(
    #                 [[1020, 437],
    #                 [1020, 665],
    #                 [604, 665],
    #                 [604,437]])
    M = cv2.getPerspectiveTransform(src, dst_point)
    M_reverse = cv2.getPerspectiveTransform(dst_point, src)
    warped = cv2.warpPerspective(img, M, img_size, flags = cv2.INTER_LINEAR)
    return warped, M, M_reverse   

# warp the unwarp result back to real image
def conners_warp(binary_warped, m_reverse):
    img_size = (binary_warped.shape[1], binary_warped.shape[0])
    unwarp = cv2.warpPerspective(binary_warped, m_reverse, img_size, flags = cv2.INTER_LINEAR)

    return unwarp

def caculate_r(yvalue, a, b):
    # Define conversions in x and y from pixels space to meters
    # ym_per_pix = 30*720 # meters per pixel in y dimension
    # xm_per_pix = 3.7*700 # meters per pixel in x dimension
    # a = a*ym_per_pix**2/xm_per_pix 
    # b = b*ym_per_pix/xm_per_pix
    # yvalue = yvalue*ym_per_pix
    return ((1 + (2*a*yvalue + b)**2)**1.5)/np.absolute(2*a)
def caculate_x_value(yvalue, fit):
    return fit[0]*yvalue**2 + fit[1]*yvalue + fit[2]
def dect_point_rectangle(win_x_low, win_y_low, win_x_high, win_y_high, nonzerox, nonzeroy):
    point_list = []
    for i in range(len(nonzerox)):
        if  win_x_low < nonzerox[i] < win_x_high and  win_y_low < nonzeroy[i] < win_y_high:
            point_list.append((nonzerox[i],nonzeroy[i]))
    return point_list

def find_lane_pixels(binary_warped, line):
    # Take a histogram of the bottom half of the image
    ysize = binary_warped.shape[0]
    histogram = np.sum(binary_warped[ysize//2 :,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    if line.first == True:
        line.left_base = leftx_base
        line.right_base = rightx_base
        line.width = rightx_base - leftx_base
        # line.first = False
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = ysize//nwindows
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    # mid_current = (leftx_base + rightx_base)

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin//2
        win_xleft_high = leftx_current + margin//2 
        win_xright_low = rightx_current - margin//2
        win_xright_high = rightx_current + margin//2 
        
        # Draw the windows on the visualization image
        # cv2.rectangle(out_img,(win_xleft_low,win_y_low), (win_xleft_high,win_y_high),(255,0,0), 2) 
        # cv2.rectangle(out_img,(win_xright_low,win_y_low), (win_xright_high,win_y_high),(0,255,0), 2) 
        ### TO-DO: Identify the nonzero pixels in x and y within the window ### 
        good_left_inds = dect_point_rectangle(win_xleft_low, win_y_low, win_xleft_high, win_y_high, nonzerox, nonzeroy)
        good_right_inds = dect_point_rectangle(win_xright_low, win_y_low, win_xright_high, win_y_high, nonzerox, nonzeroy)
        
        # Append these indices to the lists
        if len(good_left_inds) > 0:
            left_lane_inds.append(good_left_inds)
        if len(good_right_inds) > 0:
            right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean([x[0] for x in good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean([x[0] for x in good_right_inds]))


    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(np.reshape(left_lane_inds, -1))
        # print(left_lane_inds)
        right_lane_inds = np.concatenate(np.reshape(right_lane_inds, -1))
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    # print(left_lane_inds)
    try:
        leftx = [point[0] for point in left_lane_inds]
        lefty = [point[1] for point in left_lane_inds]
        rightx = [point[0] for point in right_lane_inds]
        righty = [point[1] for point in right_lane_inds]
    except:
        leftx = []
        lefty = []
        rightx = []
        righty = []
    if line.first == True:
        try:
            line.left_fit = np.polyfit(lefty, leftx, 2)
            line.right_fit = np.polyfit(righty, rightx, 2)
            line.best_fit_left = line.left_fit
            line.best_fit_right = line.right_fit
            line.wide_base = abs(line.left_fit[2] + line.right_fit[2])/2
            # line.first = False
        except:
            # line.first = True
            pass

    return leftx, lefty, rightx, righty
    # return out_img, nonzero
def generate_fit_line(y_top, y_bottom, left_fit, right_fit):
    ploty = np.linspace(y_bottom, y_top, y_top - y_bottom + 1)
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty    

    return left_fitx, right_fitx, ploty

def fit_poly(binary_warped, leftx, lefty, rightx, righty, line):
    yvalue = binary_warped.shape[0]
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # ym_per_pix = 1
    # xm_per_pix = 1
    # leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # left_fit_r = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        # right_fit_r = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        
    except (ValueError, TypeError):
        left_fit = np.array([ 1, 1, 1])
        right_fit = np.array([1, 1, 1])
        # left_fit_r = np.array([ 1, 1, 1])
        # right_fit_r = np.array([1, 1, 1])

    try:
        left_fit_r = np.polyfit(np.float32(lefty)*ym_per_pix, np.float32(leftx)*xm_per_pix, 2)
        right_fit_r = np.polyfit(np.float32(righty)*ym_per_pix, np.float32(rightx)*xm_per_pix, 2)
        Rleft = np.int32(caculate_r(np.float32(yvalue)*ym_per_pix, left_fit_r[0], left_fit_r[1]))
        Rright = np.int32(caculate_r(np.float32(yvalue)*ym_per_pix, right_fit_r[0], right_fit_r[1]))
    except (TypeError, ValueError):
        Rleft = 0
        Rright = 0

    # Generate x and y values for plotting
    # Rleft = np.int32(caculate_r(yvalue, left_fit[0], left_fit[1]))
    # Rright = np.int32(caculate_r(yvalue, right_fit[0], right_fit[1]))
    # Rleft = np.int32(caculate_r(yvalue*ym_per_pix, left_fit_r[0], left_fit_r[1]))
    # Rright = np.int32(caculate_r(yvalue*ym_per_pix, right_fit_r[0], right_fit_r[1]))
    # if(line.first == True):
    if (abs(Rleft - 1000) < abs(line.last_radius[0] -1000)):
        line.last_radius[0] = Rleft
        line.best_fit_left = left_fit
    if (abs(Rright - 1000) < abs(line.last_radius[1] -1000)):
        line.last_radius[1] = Rright
        line.best_fit_right = right_fit

    line.confidence_r = Rright/1000
    line.confidence_l = Rleft/1000
    
    # check the parawall
    diff_fit = np.array(left_fit) - np.array(right_fit)
    diff = np.mean(diff_fit)
    current_middle_base =  0
    # check the middle point of line position
    x1 = caculate_x_value(yvalue, left_fit)
    x2 = caculate_x_value(yvalue, right_fit)
    current_middle_base = abs(x1 + x2)/2
    if diff > 2:
        left_fit = line.best_fit_left
        right_fit = line.best_fit_right
    else:        
        # line.wide_base = abs(x1 + x2)/2
        if current_middle_base > 600 or current_middle_base < 500:
            left_fit = line.best_fit_left
            right_fit = line.best_fit_right


        
        

    # check the line curvature
    if 0.5 < line.confidence_l < 1.5:
        #line.last_radius[0] = Rleft 
        line.left_fit = left_fit
        line.recent_fit_left.append(left_fit)
        if len(line.recent_fit_left) > 10:
            line.recent_fit_left.pop(0)
    else:
        left_fit = line.best_fit_left
    if 0.5 < line.confidence_r < 1.5:
        #line.last_radius[1] = Rright
        line.right_fit = right_fit  
        line.recent_fit_right.append(right_fit)
        if len(line.recent_fit_right) > 10:
            line.recent_fit_right.pop(0)
    else:
        right_fit = line.best_fit_right

   
    if len(line.recent_fit_left) != 0 and len(line.recent_fit_right) != 0:
        fit_r = np.array(line.recent_fit_right)
        fit_l = np.array(line.recent_fit_left)
        left_fit = np.mean(fit_l, axis = 0)
        right_fit = np.mean(fit_r, axis= 0)
    text1 = "Left Radius: " + str(Rleft) + "  Right Radius: " + str(Rright)
    text2 = "confidence: r, " + str(line.confidence_r) + "  l," + str(line.confidence_l) + \
        "  line_minddle base: " + str(current_middle_base)
    # ploty = np.linspace(binary_warped.shape[0] - ytop -1, binary_warped.shape[0]-1, ytop)

    left_fitx, right_fitx, ploty = generate_fit_line(yvalue, 0, left_fit, right_fit)   


    return left_fitx, right_fitx, ploty, text1, text2

# find points in defined region
def dect_point_region(nonzero, margin, fit):
    point = []
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    for i in range(len(nonzeroy)):
        x_low = fit[0]*nonzeroy[i]**2 + fit[1]*nonzeroy[i] + fit[2] - margin//2
        x_high = x_low + margin
        if x_low < nonzerox[i] < x_high:
            point.append((nonzerox[i], nonzeroy[i]))
    return point


def search_around_poly(binary_warped, line):
    # HYPERPARAMETER
    if 0.5 <line.confidence_r < 1.5:
        margin_r = 50
    else:
        margin_r = 100 
    if 0.5 <line.confidence_l < 1.5:
        margin_l = 50 
    else:
        margin_l = 100
    
    if line.first == True:
        leftx, lefty, rightx, righty = find_lane_pixels(binary_warped, line)
        line.first = False        
    else:
        # Grab activated pixels
        nonzero = binary_warped.nonzero()    
        
        left_lane_inds = dect_point_region(nonzero, margin_l, line.left_fit)
        right_lane_inds = dect_point_region(nonzero, margin_r, line.right_fit)
        if len(left_lane_inds) > 1000 and len(right_lane_inds) > 1000:
            # Again, extract left and right line pixel positions
            leftx = [point[0] for point in left_lane_inds]
            lefty = [point[1] for point in left_lane_inds]

            rightx = [point[0] for point in right_lane_inds]
            righty = [point[1] for point in right_lane_inds]
        else:
            leftx, lefty, rightx, righty = find_lane_pixels(binary_warped, line)

    # Fit new polynomials
    left_fitx, right_fitx, ploty, text1, text2 = fit_poly(binary_warped, leftx, lefty, rightx, righty, line)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img) 
    try:
        window_img[lefty, leftx] = [255, 0, 0]
        window_img[righty, rightx] = [0, 255, 0]  
    except:
        pass

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    line_margin = 20
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-line_margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+line_margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    # print(len(left_line_pts))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-line_margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+line_margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    
    left_line_window = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line_window = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    region_pts = np.hstack((left_line_window, right_line_window))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,0, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))

    # cv2.fillPoly(window_img, np.int_([region_pts]), (0, 255, 255))
    
    # Plot the polynomial lines onto the image
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
    return window_img, text1, text2, leftx


def weighted_img(img, initial_img, α=0.8, β=3., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

# Always test! 
# test of colorspace
straight1 = "./test_images/straight_lines1.jpg"
straight2 = "./test_images/straight_lines2.jpg"
test1 = "./test_images/test1.jpg"
test2 = "./test_images/test2.jpg"
test3 = "./test_images/test3.jpg"
test4 = "./test_images/test4.jpg"
test5 = "./test_images/test5.jpg"
test6 = "./test_images/test6.jpg"

# read in the saved camera matrix and distortion coefficients
dist_src = "my_output/wide_dist_pickle.p"
dist_pickle = pickle.load(open(dist_src, "rb"))
mtx = dist_pickle["mtx"] # camera matrix
dist = dist_pickle["dist"] # distortion coefficient
image = mpimg.imread(straight1)
ysize = image.shape[0]
xsize = image.shape[1]
print((xsize, ysize))
undistort = Camera_cali.undistort(image, mtx, dist)
gray = cv2.cvtColor(undistort, cv2.COLOR_RGB2GRAY)


# region mask
# define the vertices of ladder-shaped
left_bottom = (50, ysize)
right_bottom = (xsize-50, ysize)
left_apex = (xsize//2 - 100, ysize//2 + 100)
right_apex = (xsize//2 + 100, ysize//2 + 100)
ytop = ysize//2 - 100
vertices = np.array([[left_bottom, right_bottom, right_apex, left_apex]], dtype = np.int32) ##why [[ ]]
region_mask = region_of_interest(undistort, vertices)

# RGB space
rgb_thresh = [(0, 200), (0, 200), (0, 50)]
rgb_mask = rgb_select(region_mask, rgb_thresh) 


# HLS space
hls_thresh = (100, 255)
hls_mask = hls_select(region_mask, hls_thresh)
color_combined = np.zeros_like(hls_mask)
color_combined[(rgb_mask == 1) | (hls_mask == 1)] = 1
# hls_mask = cv2.cvtColor(hls_mask, cv2.color_hls2gray)
# # print(hls_mask)

# SOBER filter -x
sober_thresh = (50, 300)
sorber_filter_x = abs_sobel_thresh(gray,'x', 15, sober_thresh)
sorber_filter_y = abs_sobel_thresh(gray,'y', 15, sober_thresh)


# sobel filter -magnitude
magnitude_thresh = (30, 150)
mag_filter = mag_thresh(gray, 9, magnitude_thresh)

# Sober filter -arctan
arctan_thresh = (0.7, np.pi/2)
sorber_filter_ran = dir_threshold(mag_filter, 15, arctan_thresh)

combined = np.zeros_like(sorber_filter_x)
# combined[(mag_filter == 1) & (sorber_filter_ran ==1)] =1
combined[(color_combined == 1) | (((mag_filter==1) | (sorber_filter_ran ==1)) & (sorber_filter_x ==1))] = 1

region_mask_final = region_of_interest(combined, vertices)
binary_warped, M, M_reverse = corners_unwarp(region_mask_final)
# leftx, lefty, rightx, righty, out_img = find_lane_pixels(region_mask_final)
line = line(mtx, dist)
out_img, text1, text2, leftx = search_around_poly(binary_warped, line)

reverse_outimg = conners_warp(out_img, M_reverse)
reverse_outimg = region_of_interest(reverse_outimg, vertices)

img_weighted = weighted_img(reverse_outimg, undistort)

# ransform perspective

# draw the line

# Polynomial fit values from the previous frame
# Make sure to grab the actual values from the previous step in your project!
# left_fit = np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02])
# right_fit = np.array([4.17622148e-04, -4.93848953e-01,  1.11806170e+03])
# result = search_around_poly(binary_warped, left_fit, right_fit)

# how to combine to image to make same part 
# reforce?? try cv2.addWeight()
# combined = np.zeros_like(hls_mask)
# combined[(rgb_mask ==1) | (hls_mask == 1)] = 1
# binary_warped, M, M_reverse = corners_unwarp(undistort)


f, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
f.tight_layout()
# ax1.imshow(combined)
ax1.imshow(undistort)
ax1.set_title('original image', fontsize = 10)
ax2.imshow(img_weighted)
ax2.set_title('back down image', fontsize = 10)

plt.show()

'''
 test of class camera_cali()
dir_dst = './camera_cal/cali*.jpg'
camera1 =Camera_cali(9,6) 
camera1.caculate_para(dir_dst)
test = './camera_cal/calibration1.jpg'
out_dst = "./my_output/wide_dist_pickle.p"
# camera1.save_as_pickle(out_dst)
image = cv2.imread(test)
undistort_img = camera1.undistort(test, camera1.mtx, camera1.dist)
write_name = 'my_output/undistort.jpg'
cv2.imwrite(write_name, undistort_img)

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('original image', fontsize = 50)
ax2.imshow(undistort_img)
ax2.set_title('undistorted image', fontsize = 50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# plt.imshow(undistort_img,cmap = 'gray')

plt.show()

'''
    
