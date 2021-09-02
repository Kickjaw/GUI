import numpy as np
import cv2

# load the image
img = cv2.imread('118A0841.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(gray, 5)

cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

print('Original Dimensions : ',img.shape)

 
scale_percent = 10 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
print('Resized Dimensions : ',resized.shape) 
 

circles = cv2.HoughCircles(resized, cv2.HOUGH_GRADIENT, 1, 120, param1 = 100, param2 = 75, minRadius = 175, maxRadius = 400)
circles = np.uint16(np.around(circles))

mask = np.zeros((height, width), np.uint8)


for	i in circles[0,:]:
	cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),thickness=-1)


mask_full_size = cv2.resize(mask, (5792,8688), interpolation = cv2.INTER_AREA)


masked_data = cv2.bitwise_and(img, img, mask=mask_full_size)

_,thresh = cv2.threshold(masked_data,1,255,cv2.THRESH_BINARY)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)
crop = masked_data[y:y+h,x:x+w]


cv2.imwrite('maskedMuscles3.jpg', crop)



#Code to close Window

cv2.waitKey()
cv2.destroyAllWindows()
