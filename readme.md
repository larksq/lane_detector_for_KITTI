# What is this for?

This rep will detect ego lane of current BEV(bird eye view) image following the convention of the KITTI UM-LANE challenge. 

This approach use no fancy machine learning or deep neural networks. All you need is basic python libs. Including:
- Opencv (no fancy functions, so all version should be compatible)
- Numpy
- SCIPY
- SKLearn

This approach is highly controllable over all complicated cases in the KITTI dataset. It worked specially good on dashed lines which is the dominant case in KITTI UM_LANE dataset. I used a sliding window to search for dashed lines. See the visualization example for more details.

# How to use

Example


# More details

Here is the full flow chart of the detection method.
![Flow Chart](readme_images/flow_chart.png)
This approach is based on the prior knowledge of:

## What is the lines? 
The predesigned classes are: 
- SS = Straight Solid line
- CD = Curved Dashed line
- CS = Curved Solid line
- SD = Straight Dashed line

## Where does the line start from?
given an approximated starting point in the image


