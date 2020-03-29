## Writeup of ADVANCED-LANE-LINES

### This is a brief describtion how the pipeline worked and how I conquered the question in the process.

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

[//]: # (Image References)

[image1]: ./my_output/undistort.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "test 1 pic"
[image3]: ./my_output/undistort_example.png "undistort result"
[image4]: ./my_output/transform3.png "image transform Example"
[image5]: ./my_output/failure_transform.png "failure transform Example"
[image6]: ./my_output/line-point-detected.PNG "line points detected"
[image7]: ./my_output/transform4.png "warp back"
[video1]: ./my_output/project_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README
>note: I implement caculating the camera martix and distortion coefficients in 'helper.py' which is all main function/class loacted. The main file <Advanced_lane_finding.py> is just constructing the pipeline with individual function/class in 'helpers.py'. 

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I defined a **class** of Camera_cali()which caculate the camera matrix and distortion coefficients with function "caculate_para()" with below steps:
* prepare a matrix of standard chessboard point(same with the training picture points' number) with depth-axis zero.
* use glob.glob() to grab all images in directory "./camera_cal" in *images*
* enumerate the images to iterate caculating all cornner points with function **cv2.findChessboardCorners()**, store these corner points in *imgpoint* 
* get the parameters of camera matrix and dixtortion coefficients with **cv.calibrateCamera()** and store them in class and "wide_dist_pickle.p" file with function **save_as_pickle**.
* then I write a test case in the end of helper.py to undistort one picture in "camera/cal" with function **undistort()** and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
This step is really simple:
* first, read the camera matrix and distortion coefficients from the pickle file "wide_dist_pickle.p"
* second, apply the undistortion function defined in **class** of **camera.undistort()** to the tested image. The result is like bellow:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
                    [[712, 464],
                    [936, 612],
                    [368, 615],
                    [572, 464]])
dst_point = np.float32(
                    [[960, 70],
                    [960, 630],
                    [320, 630],
                    [320, 70]]))
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 712, 464      | 960, 70        | 
| 936, 612      | 960, 630      |
| 368, 615     | 320, 720      |
| 572, 464      | 320, 70        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
And I also tried a src points and dst points as below which confused me at first place:
```python
src = np.float32(
                    [[712, 464],
                    [936, 612],
                    [368, 615],
                    [572, 464]])
dst_point = np.float32(
                    [[712, 464],
                    [712, 630],
                    [572, 630],
                    [572, 464]]))
```
Then I got the result as below:

![failure transform example][image5]

I also apply a region mask agfer take such tranformation. But the filtered road line is very small at first place. But I think this kind of transform having its' advantage: the Curvature of lane will be more accurate( cause the lane points is compressed in a small region). But the disadvantage of this transformation is the case I can't detect point at some windows at all. So I think the first points given is a better choice!



#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image6]
Basicly, I identify these point in transformed binary_warped image by sliding windows as shown in class.
1. caculate the base point of each line at bottom of the image with 'histogram method'
2. define the slide windows size, and find the point in this windows. then iterate this method by update(moving) the base point of windows in each layer(if the points in windows is more than defined threhold the caculate the center x of these point as updated base point. Ohterwize ,keep the base points as last one). These steps are difined in **find_lane_pixels**, I alse defined a function **dect_point_rectangle** to he;p identifing whether point nonezero is in current window box.
3. fit the points identifed last step with **np.polyfit(pointy, pointx, 2)**, this function will return the coefficient of line of 'x = a*x^2 + b*x + c'. Then I try to genarate a series of points: y with **np.linspace** and resposed x with coefficient. Then add these point in binary_warped image.
#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # 235 # in function **caculate_r** , this function will return the curvature of bottom line of left and right lane using given formulation. And I returned the curvature info and add it in image in 'Advanced_lane_fiding.py' of line # 86 # to display it.
And this caculated curvature of line is in the real worlf cause I get the fit parameter with code below in **fit_ploy()** function:
```python
    yvalue = binary_warped.shape[0]
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    left_fit_r = np.polyfit(np.float32(lefty)*ym_per_pix, np.float32(leftx)*xm_per_pix, 2)
    right_fit_r = np.polyfit(np.float32(righty)*ym_per_pix, np.float32(rightx)*xm_per_pix, 2)
    Rleft = np.int32(caculate_r(np.float32(yvalue)*ym_per_pix, left_fit_r[0], left_fit_r[1]))
    Rright = np.int32(caculate_r(np.float32(yvalue)*ym_per_pix, right_fit_r[0], right_fit_r[1]))
```
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # 89 # in my code in `Advanced_lane_finding.py` in the function `conners_warp()` in `helper_p2.py`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [vedio1](./my_output/project_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

[image8]: ./my_output/failure_transform.png "transform failure"
[image9]: ./my_output/failure-3.png "siding window failure"
[image10]: ./my_output/pipeline.png "my pipe line"
Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
1. problem or issues：
    * fail to detect the line by using a histogram and sliding window. This is beacause i forgot that I have already transform the image to "eagle view", then I transfer the worng window margin.
![alt text][image9]
    * fail to warp the image to proper shape, the transformed line only take small portion of the binary_image as below. I sovled  this by select a larger region matriax of the image.
![alt text][image8]
    * fail to transfer the `line()` class to the image-process funcitons. I solved this by define a global variant in line # 130 # and # 20 #.
2. my approach:
![alt text][image10]
The picture up is the pipeline expreed by flows when I processes vedio(process image with distortion and apply fillters is simplify). For now the pipeline works well with `project_vedio.mp4` while having some flaws with chanllenge videos which I will work on later. 
3. possible improvement ways:
   * conbining the dection of line at the bottom of image
   * tuning parameter to get better  binary_warped image 
   * add a filter of h channel of hsv image(avoid shade)
   * rebuild a confidence function representing how the lines fited. 