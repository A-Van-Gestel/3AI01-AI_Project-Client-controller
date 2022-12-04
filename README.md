# 3AI01-AI_Project-Client-controller

## Getting started
### Requirements
- Python 3.10

### Project setup
1. Create a Python venv: `python -m venv venv`
	
2. Activate the venv using ONE of the following commands
	- CMD: `.\venv\Scripts\Activate.cmd`
	- PowerShell: `.\venv\Scripts\Activate.ps1`
	
3. Install the dependencies: `pip install -r requirements.txt`

4. Run the code:
	- `py .\drive.py` in a console (**recommended**)
	- Or run it in your editor of choice


## Scripts
### Main scripts
- Main client program: `drive.py`
- OpenCV - Average lanes: `road_lane_detection.py`
  - Based on: [Building a lane detection system
with Python 3 & OpenCV](https://medium.com/analytics-vidhya/building-a-lane-detection-system-f7a727c6694)


### Test scripts
- OpenCV - Find best canny filter values: `opencv.py`

### Broken or old scripts
- OpenCV - Average lanes (Broken): `opencv_lane_detection_simple.py`
  - Based on: [d-misra/Lane-detection-opencv-python
](https://github.com/d-misra/Lane-detection-opencv-python/blob/master/Lane-detection-opencv.ipynb)
- OpenCV - Best lanes only: `opencv_lane_detection.py`
  - Based on: [Finding Lane Lines on the Road
](https://jefflirion.github.io/udacity_car_nanodegree_project01/P1.html)


### Information
- [GitHub issue: Connecting with the Unity project](https://github.com/udacity/self-driving-car-sim/issues/131)