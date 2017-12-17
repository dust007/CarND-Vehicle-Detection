## Vehicle Detection
### Xiangjun Fan

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # "Image References"
[image1]: ./output_images/car_noncar.png
[image2]: ./output_images/hog.png
[image3]: ./output_images/hot_win1.png
[image4]: ./output_images/hot_win2.png
[image5]: ./output_images/hot_win3.png
[image6]: ./output_images/hot_win4.png
[image7]: ./output_images/label1.png
[image8]: ./output_images/label2.png
[image9]: ./output_images/label3.png
[image10]: ./output_images/label4.png
[video1]: ./project_video_output3.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cell `Lesson functions` of the IPython notebook, with function name `get_hog_features()`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Here is an example using the `gray` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters.

*  `color_space='RGB'`, `orient=8`, `pix_per_cell=8`, `cell_per_block=2`, `hog_channel=0`. SVM result is good, but windows on test image can not cover the whole car.
*  Tried `color_space='LUV'`, `color_space='HLS'`, `color_space='HSV'`, similar result.
*  Tried  `orient=8`, `orient=10`, `orient=12`, change is minor.
*  Think of using all channels instead of channel 0, turns out it works better. However, it has more features which make the detection slower. Finally choose`color_space='RGB'`, `orient=8`, `pix_per_cell=8`, `cell_per_block=2`, `hog_channel="ALL"`. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using all spatial feature, hist feature and hog features. It is in code cell `Train SVC classifier` of the IPython notebook.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions using `slide_window()` with multiple `xy_window_sizes=[(80,80), (120,120), (160, 160)]`, `xy_overlap=(0.75, 0.75)`, `y_start_stop = [360, 600]` of the IPython notebook.

* I first tried `xy_window_sizes=[(96,96)]`,  `xy_overlap=(0.5, 0.5)`, but it misses small cars(cars far way).
* Then I tried  `xy_window_sizes=[(64,64), (96,96)]`,  `xy_overlap=(0.5, 0.5)`, but it captures too much false positives.
* Then I change the window scale to  `xy_window_sizes=[(80,80), (120,120), (160,160)]`,  `xy_overlap=(0.5, 0.5)`. This time it captures both small and bit cars, however the overlap is not enough to have more boxes around car. Therefore overlap was later increased to 0.75.

#### 2. Show some examples of test images to demonstrate how your pipeline is working. 

Here are some example images:

![alt text][image3]

![alt text][image4]

---
### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are the frames before and after filtering:

* Before
  ![alt text][image3]
* After
  ![alt text][image7]
* Before
  ![alt text][image4]
* After
  ![alt text][image8]
* Before
  ![alt text][image5]
* After
  ![alt text][image9]
* Before
  ![alt text][image6]
* After
  ![alt text][image10]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image8]

---
### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* The major issue I faced was false positive and car box boundary. 
  * I have tried quite a few HOG parameters and choose best one by looking at SVM accuracy and test image performance.
  * With the hint from project review, to store heatmap of previous frames will dramatically help both reduce false positive and smooth car boxes.
  * Maybe combine multiple HOG feature for SVM will help too.
  * Could use a better classifier, such as SVM with nonlinear kernel, or boosting tree model.
* The video detection efficiency is also a concern.
  * Could improve HOG calculation using the technique in the course, although no time to experiment it.
  * Could continue to explore HOG parameter, and/or reduce SVM feature vector size to increase the detection efficiency.
  * Could do parallelization on frame process to reduce video process time. 