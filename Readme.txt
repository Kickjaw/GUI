coding standards
methods -> mMethod
objects -> oObject




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



things to add:
circle masking
file load blocking for not acceptable images
exception handling
edge case
error logging
error reporting to email system



Notes:


V1

end user open the image, click button, identify mussels, take measurements, output image and table to files

different color bounding boxes per class

options for end user:
batch processing
toggle detection boxes
- toggle by mussel type
input pixel to inch scaling factor



measurements: 
length, width, hinge, aligned to muscle

error log

output table: importable into spreadsheet, common format, csv or such
flag errors

table id, mussel ID, length, width, hinge, 

build as .exe or universal exacutable




V2
interactable bounding boxs that effect table
alternative setups
alternative resolutions
automatic sclaing factor input with known dimensional target
