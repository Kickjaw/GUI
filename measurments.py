#https://www.uniquesoftwaredev.com/calculating-the-size-of-objects-in-photos-with-computer-vision/
#conda install -c conda-forge scikit-learn 
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



image = cv2.imread('6.png')
image = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2), interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7,7), 0)


edge_detect = cv2.Canny(gray, 15, 100)
edge_detect = cv2.dilate(edge_detect, None, iterations=1)
edge_detect = cv2.erode(edge_detect, None, iterations=1)

cntours = cv2.findContours(edge_detect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cntours = imutils.grab_contours(cntours)

(cntours, _) = contours.sort_contours(cntours)

area, triangle = cv2.minEnclosingTriangle(cntours[0])
(x,y),radius = cv2.minEnclosingCircle(cntours[0])
center = (int(x),int(y))
radius = int(radius)



blank = np.zeros(image.shape[0:2])
img1 = blank.copy()
cv2.circle(img1,center,radius,(255,255,255),1)
img2 = cv2.drawContours(blank.copy(), cntours, 0, 1)
intersection = np.logical_and(img1, img2)
intersection = np.transpose(np.nonzero(intersection))

k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
k_means.fit(intersection)
k_means_cluster_centers = k_means.cluster_centers_

for i in range(3):
    cv2.line(image, tuple(k_means_cluster_centers[i]), tuple(k_means_cluster_centers[(i+1)%3]), (255, 255, 0), 1)

plt.imshow(image, cmap='gray')
plt.show()


def normalizeAngle(triangle):
    P1 = triangle[0][0]
    P2 = triangle[1][0]
    P3 = triangle[2][0]

    A1 = np.arccos(np.dot(P1-P2, P1-P3)/(np.linalg.norm(P1-P2)*np.linalg.norm(P1-P3)))
    A2 = np.arccos(np.dot(P2-P1, P2-P3)/(np.linalg.norm(P2-P1)*np.linalg.norm(P2-P3)))
    A3 = np.arccos(np.dot(P3-P1, P3-P2)/(np.linalg.norm(P3-P1)*np.linalg.norm(P3-P2)))
    
    if A1 == min(A1, A2, A3):
        rotationAngle = -np.arctan((P2[1]-P3[1])/(P2[0]-P3[0])) #y/x
    if A2 == min(A1, A2, A3):
        rotationAngle = -np.arctan((P1[1]-P3[1])/(P1[0]-P3[0])) #y/x
    if A3 == min(A1, A2, A3):
        rotationAngle = -np.arctan((P2[1]-P1[1])/(P2[0]-P1[0])) #y/x
    print(rotationAngle)
    return rotationAngle



for i in range(3):
    cv2.line(image, tuple(triangle[i][0]), tuple(triangle[(i+1)%3][0]), (255, 255, 0), 1)


#image = imutils.rotate_bound(image,np.degrees(normalizeAngle(triangle)))


plt.imshow(image)
plt.show()


""" 

pixel_to_size = 1
# function for finding the midpoint
def mdpt(A, B):
    return ((A[0] + B[0]) * 0.5, (A[1] + B[1]) * 0.5)


for c in cntours:
    if cv2.contourArea(c) < 100: #ignore/fly through conturs that are not big enough 
        continue
    # compute the rotated bounding box of the contour; should handle cv2 or cv3..
    orig = image.copy()
    bbox = cv2.minAreaRect(c)
    bbox = cv2.cv.boxPoints(bbox) if imutils.is_cv2() else cv2.boxPoints(bbox)
    bbox = np.array(bbox, dtype="int")
    # order the contours and draw bounding box
    bbox = perspective.order_points(bbox)
    cv2.drawContours(orig, [bbox.astype("int")], -1, (0, 255, 0), 2)

    for (x, y) in bbox:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255),-1)
        # unpack the ordered bounding bbox; find midpoints
    (tl, tr, br, bl) = bbox
    (tltrX, tltrY) = mdpt(tl, tr)
    (blbrX, blbrY) = mdpt(bl, br)
    (tlblX, tlblY) = mdpt(tl, bl)
    (trbrX, trbrY) = mdpt(tr, br)

    # draw the mdpts on the image (blue);lines between the mdpts (yellow)
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0,0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX),
    int(blbrY)),(0, 255, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX),
    int(trbrY)),(0, 255, 255), 2)
    # compute the Euclidean distances between the mdpts
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    distA = dA
    distB = dB
    # draw the object sizes on the image
    cv2.putText(orig, "{:.1f}Px".format(distA),(int(tltrX - 10), int(tltrY - 10)), cv2.FONT_HERSHEY_DUPLEX,0.55, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f}Px".format(distB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_DUPLEX,0.55, (255, 255, 255), 2)


# show the output image
plt.imshow(orig)
plt.show()
 """