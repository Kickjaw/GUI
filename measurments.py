#https://www.uniquesoftwaredev.com/calculating-the-size-of-objects-in-photos-with-computer-vision/
#conda install -c conda-forge scikit-learn 
#conda install -c conda-forge imutils
#conda install -c conda-forge opencv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time as time
from imutils import perspective
from imutils import contours
import imutils
from numpy.lib.arraysetops import intersect1d
from scipy.spatial import distance as dist
from sklearn.cluster import KMeans
from math import copysign, log10

white = (255,255,255)
master = cv2.imread('9.png')
image1 = cv2.imread('6.png')
src = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('4.png')
images = [master, image1, src, image3]
scaledImages = []
cts=[]
x = []

kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernal2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
kernal3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))

def circleMask(image):
    center = (int(image.shape[1]/2),int(image.shape[0]/2))
    mask = np.zeros(image.shape[0:2], np.uint8)
    cv2.circle(mask, center, 15,255,-1)
    return mask

# Thresholding using THRESH_BINARY_INV 
th, dst = cv2.threshold(src,127,255, cv2.THRESH_BINARY_INV)
x = cv2.erode(dst, kernal3, iterations=3)
y = cv2.bitwise_and(x, circleMask(x))
z = cv2.dilate(y, kernal3, iterations=2)
h = cv2.bitwise_and(src,src, mask = z)


 
plt.subplot(2,2,1), plt.imshow(h, cmap='gray')
plt.subplot(2,2,2), plt.imshow(src, cmap='gray')
plt.subplot(2,2,3), plt.imshow(y, cmap='gray')
plt.subplot(2,2,4), plt.imshow(z, cmap='gray')
plt.show()

# Thresholding using THRESH_TRUNC 
th, dst = cv2.threshold(src,127,255, cv2.THRESH_TRUNC); 











""" 
def HuMoments():
    image = cv2.imread('6.png', cv2.IMREAD_GRAYSCALE)
    _,binImage = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    # Calculate Moments 
    moments = cv2.moments(binImage) 
    # Calculate Hu Moments 
    huMoments = cv2.HuMoments(moments)
    for i in range(0,7):
        huMoments[i] = -1* copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))


def getContours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edge_detect = cv2.Canny(gray, 15, 75)
    edge_detect = cv2.dilate(edge_detect, kernal3, iterations=1)
    edge_detect = cv2.erode(edge_detect, kernal3, iterations=1)

    cntours = cv2.findContours(edge_detect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cntours = imutils.grab_contours(cntours)
    cv2.drawContours(image,cntours,-1,white,1)
    plt.imshow(image)
    plt.show()
    (cntours, _) = contours.sort_contours(cntours)
    return cntours

#shape matching tests
for image in images:
    scaledImages.append(cv2.resize(image, (image.shape[1]*2, image.shape[0]*2), interpolation=cv2.INTER_CUBIC))

for i in range(1,5):
    plt.subplot(2,2,i), plt.imshow(scaledImages[i-1], cmap='gray')

plt.show()

for image in scaledImages:
    cts.append(getContours(image))




for i in cts:
    img = np.zeros(scaledImages[0].shape[0:2])
    cv2.drawContours(img, i, -1,(255,255,255),thickness=cv2.FILLED)
    x.append(img)

for i in range(1,5):
    plt.subplot(2,2,i), plt.imshow(x[i-1], cmap='gray')
    print(cv2.matchShapes(x[0], x[i - 1], 1, 0.0))
plt.show()












(x,y),radius = cv2.minEnclosingCircle(cntours[0])
center = (int(x),int(y))
radius = int(radius)


blank = np.zeros(image.shape[0:2])
img1 = blank.copy()
cv2.circle(img1,center,radius,(255,255,255),1)
img2 = cv2.drawContours(blank.copy(), cntours, 0, 1)
intersection = np.logical_and(img1, img2)
combined = np.logical_or(img1, img2)

intersection = np.transpose(np.nonzero(intersection))

k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
k_means.fit(intersection)
cCenters = k_means.cluster_centers_.astype(int)
print(cCenters)
corners = []
for point in cCenters:
    corners.append([point[1], point[0]])

for i in range(3):
   cv2.line(image, tuple(corners[i]), tuple(corners[(i+1)%3]), (255, 255, 0), 1)



def normalizeAngle(corners):
    distances = {}
    distances[dist.euclidean(corners[0],corners[1])] = [corners[0],corners[1]]
    distances[dist.euclidean(corners[1],corners[2])] = [corners[1],corners[2]]
    distances[dist.euclidean(corners[2],corners[0])] = [corners[2],corners[0]]
    
    points = distances[min(distances)]

    slope = (points[0][1]-points[1][1])/(points[0][0]-points[1][0])

    angle = np.arctan(slope)

    return angle
    

image = imutils.rotate_bound(image,np.degrees(normalizeAngle(corners))-90)


cntours = getContours(image)


#define a set of Hu moments around a base mussel shape and use that to compare against to pick out
#the contour that mose closely matches the hu moments. also impement a miminum cut off 
# to fail an image if it cant fine the mussel contour




#need something to pick the mussel contour out of the rotated image
for c in cntours:
    print(cv2.contourArea(c))

areaSorted = sorted(cntours, key = cv2.contourArea, reverse=True)



cv2.drawContours(image,areaSorted[2],-1,(255,255,255),1)



pixel_to_size = 1
# function for finding the midpoint
def mdpt(A, B):
    return ((A[0] + B[0]) * 0.5, (A[1] + B[1]) * 0.5)



orig = image.copy()
x,y,w,h = cv2.boundingRect(areaSorted[2])
# order the contours and draw bounding box
cv2.rectangle(orig, (x,y), (x+w, y+h), (0, 255, 0), 2)

(tl, tr, br, bl) = ((x,y),(x+w,y),(x,y+h),(x+w,y+h))
(tltrX, tltrY) = mdpt(tl, tr)
(blbrX, blbrY) = mdpt(bl, br)
(tlblX, tlblY) = mdpt(tl, bl)
(trbrX, trbrY) = mdpt(tr, br)



distA = w
distB = h
# draw the object sizes on the image
cv2.putText(orig, "{:.1f}Px".format(distA),(int(tltrX - 10), int(tltrY - 10)), cv2.FONT_HERSHEY_DUPLEX,0.55, (255, 255, 255), 2)
cv2.putText(orig, "{:.1f}Px".format(distB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_DUPLEX,0.55, (255, 255, 255), 2)
 """