import cv2

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance as dist
import dataclasses
import inspect
from dataclasses import dataclass, field
import csv
import torch



class yolov5Detection():

    def __init__(self):
        self.image = Image.new('RGB', (1,1))
        self.imagecv2 = np.zeros((1,1,3), np.uint8)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', force_reload=True, path='C:/Users/jared/Documents/GitHub/GUI/Project Files/best.pt')
        self.detections = 0
        self.confidenceCutOff = .90
        self.musselCount = 0
        self.openMusselCount = 0
        self.fullOpenMusselCount = 0
    
    def mDetection(self):
        results = self.model(self.image, size=1280)
        self.detections = results.pandas().xyxy[0]

    def mDisplayDetections(self):
        self.musselCount = 0
        self.openMusselCount = 0
        self.fullOpenMusselCount = 0
        ConfidentDetections = self.detections[self.detections["confidence"] > self.confidenceCutOff]
        for index, row in ConfidentDetections.iterrows():
            if row["class"] == 0:
                self.musselCount +=1
                cv2.rectangle(self.imagecv2, (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])), color=(255, 255, 255), thickness=1)
            elif row["class"] == 1:
                self.openMusselCount +=1
                cv2.rectangle(self.imagecv2, (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])), color=(0, 255, 0), thickness=1)
            elif row["class"] == 2:
                self.openMusselCount +=1
                cv2.rectangle(self.imagecv2, (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])), color=(0, 0, 0), thickness=1)

    

    





    """

    #takes all the tiles and returns all the bounding boxes
    def mDetectMussels(self):
        print('detecting')
        #run the detection

        classIds, scores, boxes = self.model.detect(self.image, confThreshold=0.6, nmsThreshold=0.5)
        #box = [xcord, ycord, boxwidth, boxheight]
        #add in for loop to go over boxes and append them to the main box list, while translating them in
        #space so that they line up with the original image
        for (classId, score, box) in zip(classIds, scores, boxes):
            self.detections.append(dMussel(classId, score, box))

    
    def mDisplayDetections(self):
        print('display')
        musselCount = 0
        open_musselCount = 0
        for mussel in self.detections:
            if mussel.classId == 0:
                musselCount +=1
                cv2.rectangle(self.image, (mussel.box[0], mussel.box[1]), (mussel.box[0] + mussel.box[2], mussel.box[1] + mussel.box[3]), color=(255, 255, 255), thickness=1)
            elif mussel.classId == 1:
                open_musselCount +=1
                cv2.rectangle(self.image, (mussel.box[0], mussel.box[1]), (mussel.box[0] + mussel.box[2], mussel.box[1] + mussel.box[3]), color=(0, 255, 0), thickness=1)
            
            
            #text = '%s: %.2f' % (self.classes[classId[0]], score)
            #cv2.putText(self.image, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,color=(0, 255, 0), thickness=2)
        im2 = self.image.copy()
        im2[:, :, 0] = self.image[:, :, 2]
        im2[:, :, 2] = self.image[:, :, 0]
        print(musselCount, open_musselCount)
        cv2.imwrite('detections.jpg', self.image)
        plt.imshow(im2)
        plt.show()
    
    #for a list of detections, iterates through and measures the mussels that are closed
    def mMeasureDetectedMussels(self):
        for mussel in self.detections:
            if mussel.classId == 0:
                cropped_mussel = self.mBBoxCrop(mussel.box)
                self.mMeasureContour(cropped_mussel)
    
    #for a given bounding box returns the portion of the image within it
    def mBBoxCrop(self, box):
        #box = [xcord, ycord, boxwidth, boxheight]
        #imagecrop[startheight:endheight, startwidth:endwidth]
        image = self.image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        return image

    #takes in a image of a single detected item and returns the height and width of it
    #things to add, rotate the mussels so they all face the same direction
    def mMeasureContour(self, image):
        image = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), 0)
        edge_detect = cv2.Canny(gray, 15, 100)
        edge_detect = cv2.dilate(edge_detect, None, iterations=1)
        edge_detect = cv2.erode(edge_detect, None, iterations=1)

        cntours = cv2.findContours(edge_detect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntours = imutils.grab_contours(cntours)

        (cntours, _) = contours.sort_contours(cntours)
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
            cv2.drawContours(orig, [bbox.astype("int")], -1, (0, 255, 0), 1)

            for (x, y) in bbox:
                cv2.circle(orig, (int(x), int(y)), 2, (0, 0, 255),-1)
            # unpack the ordered bounding bbox; find midpoints
            (tl, tr, br, bl) = bbox
            (tltrX, tltrY) = mdpt(tl, tr)
            (blbrX, blbrY) = mdpt(bl, br)
            (tlblX, tlblY) = mdpt(tl, bl)
            (trbrX, trbrY) = mdpt(tr, br)

            # draw the mdpts on the image (blue);lines between the mdpts (yellow)
            # cv2.circle(orig, (int(tltrX), int(tltrY)), 2, (255, 0,0), -1)
            # cv2.circle(orig, (int(blbrX), int(blbrY)), 2, (255, 0, 0), -1)
            # cv2.circle(orig, (int(tlblX), int(tlblY)), 2, (255, 0, 0), -1)
            # cv2.circle(orig, (int(trbrX), int(trbrY)), 2, (255, 0, 0), -1)
            # cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX),
            # int(blbrY)),(0, 255, 255), 2)
            # cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX),
            # int(trbrY)),(0, 255, 255), 2)
            # compute the Euclidean distances between the mdpts
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            distA = dA
            distB = dB
            print(distA,distB)
            # draw the object sizes on the image
            #cv2.putText(orig, "{:.1f}in".format(distA),(int(tltrX - 10), int(tltrY - 10)), cv2.FONT_HERSHEY_DUPLEX,0.55, (255, 255, 255), 2)
            #cv2.putText(orig, "{:.1f}in".format(distB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_DUPLEX,0.55, (255, 255, 255), 2)
            plt.imshow(orig)
            plt.show()


    def saveDataTable(self):
        #write all the data saved in self.detections to a csv file
        filename = 'musseldatatable.csv'

        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['classId', 'Height', 'Width'])
            for mussel in self.detections:
                csvwriter.writerow([mussel.classId, mussel.height, mussel.width])
       """ 

    

    

