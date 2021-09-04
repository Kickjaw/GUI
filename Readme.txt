coding standards
methods -> mMethod
objects -> oObject
dataclass -> dClass



measuring objects in images
https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/

need to tile input images

https://stackoverflow.com/questions/38235643/getting-started-with-tensorflow-split-image-into-sub-images

yolov4 weights in python
https://dsbyprateekg.blogspot.com/2020/08/how-to-use-opencv-python-with-darknets.html

https://www.youtube.com/watch?v=FjyF03uawsA

deep nueral networks in opencv

https://docs.opencv.org/3.4/d6/d0f/group__dnn.html#ga138439da76f26266fdefec9723f6c5cd
https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/#what-is-opencvv-dnn-module
https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

building as .exe
https://stackoverflow.com/questions/12059509/create-a-single-executable-from-a-python-project



things to add:
circle masking
file load blocking for not acceptable images
exception handling
edge case
error logging
event logging
data table creatation
measuring of mussels




Notes:


V1

UI
load an image
pan/zoom on image
cirlce mask cut

display detections
    easily distinguish between classes
    toggle on/off
display count
input pixel to inch scaling factor
scrolling table on the right of the detections

DATA
total count of differnt classes
    mussel
    mussel_open
    mussel_half
measurements: 
    length
    width
    aligned to muscle
flag bad detections by measurements?

OUTPUT
table of measurements
    ID
    mussel class
    length
    width
    flag for bad data?
error log
event log


build as .exe or universal exacutable




V2
interactable bounding boxs that effect table
alternative setups
alternative resolutions
automatic sclaing factor input with known dimensional target
measurements - add hinge
