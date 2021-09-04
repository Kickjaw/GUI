import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils import perspective
from imutils import contours
import imutils
from scipy.spatial import distance as dist
import dataclasses
import inspect
from dataclasses import dataclass, field

@dataclass(order=True)
class dMussel:
    classID: int
    classname: str = ""
    boundingBox: list(int) = field(default_factory=list)



class darkentDetection(object):

    def __init__(self):
        self.image = cv2.imread('3.jpg')
        self.tilesize = 512
        self.overlap = 10
        self.classes = ['mussel', 'mussel_open']
        self.net = cv2.dnn.readNetFromDarknet('yolov4-obj-tilled.cfg', 'yolov4-obj-tilled_best.weights')
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(scale=1 / 255, size=(512, 512), swapRB=True)
        self.tiles = [] #list of tiles, tile = [tilledImage, tileYcord, tileXcord]
        self.tilledDimensions = [0,0] #[numOfXtiles, numOfYtiles]
        self.detections = [] #list of detected items, = []

    #takes in an image and overlap and returns a list of images
    #returnes list of images all the same size, if the cut tile is to small pad with black pixels
    def mTileImage(self):
        self.tilledDimensions[0] = int(np.floor(self.image.shape[1]/(self.tilesize-self.overlap)))
        self.tilledDimensions[1] = int(np.floor(self.image.shape[0]/(self.tilesize-self.overlap)))
        print(self.tilledDimensions)

        for i in range(self.tilledDimensions[1]):
            for j in range(self.tilledDimensions[0]):
                #imagecrop[startheight:endheight, startwidth:endwidth]
                #if the top left corner
                if j == 0 and i == 0:
                    tempImage = self.image[0:self.tilesize,0:self.tilesize]
                #if left most image
                elif j ==0:
                    tempImage = self.image[(self.tilesize)*i-self.overlap:(self.tilesize)*(i+1)-self.overlap,0:self.tilesize]
                #if in top row
                elif i ==0:
                    tempImage = self.image[0:self.tilesize,(self.tilesize)*j-self.overlap:(self.tilesize)*(j+1)-self.overlap]
                else:
                    tempImage = self.image[(self.tilesize*i)-self.overlap:(self.tilesize)*(i+1)-self.overlap,(self.tilesize*j)-self.overlap:(self.tilesize)*(j+1)-self.overlap]


                #pad the image with blackness if it not large enough to be tile sized
                if tempImage.shape[1] != self.tilesize:
                    tempImage = cv2.copyMakeBorder(tempImage, 0, 0, 0, (self.tilesize-tempImage.shape[1]),cv2.BORDER_CONSTANT,value=[0,0,0])
                if tempImage.shape[0] != self.tilesize:
                    tempImage = cv2.copyMakeBorder(tempImage, 0, (self.tilesize-tempImage.shape[0]), 0, 0, cv2.BORDER_CONSTANT,value=[0,0,0])
                self.tiles.append([tempImage, i, j])

    #takes all the tiles and returns all the bounding boxes
    def mDetectMussels(self):
        print('detecting')
        #run the detection
        for tile in self.tiles:
            #maybe run the detection with out any of the non max suppresion, then combine all the boxes, then run the nonmax supression
            classIds, scores, boxes = self.model.detect(tile[0], confThreshold=0.6, nmsThreshold=0.5)
            #box = [xcord, ycord, boxwidth, boxheight]
            #add in for loop to go over boxes and append them to the main box list, while translating them in


            #space so that they line up with the original
            for (classId, score, box) in zip(classIds, scores, boxes):
                # collisionsToUse = self.mEdgesToUse(tile)
                # if self.mEdgeCollision(box, collisionsToUse):
                    if tile[1] == 0 and tile[2] == 0:
                        boxTranslated = box
                    elif tile[2] == 0:
                        boxTranslated = [box[0], box[1]+(tile[1]*512-self.overlap), box[2], box[3]]
                    elif tile[1] == 0:
                        boxTranslated = [box[0]+(tile[2]*512-self.overlap), box[1], box[2], box[3]]
                    else:
                        boxTranslated = [box[0]+(tile[2]*512-self.overlap), box[1]+(tile[1]*512-self.overlap), box[2], box[3]]
                    #change this it numpy array
                    
                    self.detections.append([classId, score, boxTranslated])

    #method to determine if dectection should be used or not with overlap
    #returns bool
    def mEdgeCollision(self, box, collisionstouse):
        #tile = [tilledImage, tileYcord, tileXcord]

        collisionLTRB = [False, False, False, False] #list of bools indicating if box is colliding with left, top, right, bottom edges of the tile
        
        #check which edges the box collides with
        #box = [xcord, ycord, boxwidth, boxheight]
        if box[0] < 2:
            collisionLTRB[0] = True
        if box[1] < 2:
            collisionLTRB[1] = True
        if box[0] + box[2] > self.tilesize-2:
            collisionLTRB[2] = True
        if box[1] + box[3] > self.tilesize-2:
            collisionLTRB[3] = True
        if any(collisionLTRB):
            if collisionLTRB == collisionstouse:
                return True
            else:
                return False
        else:
            return True

    #method to check which detections to use based on tile location
    #returns a list of 4 bools, left, top, right, bottom, of wether to use that edge or not
    def mEdgesToUse(self, tile):
        #check which edges the current tile will use
        #if in the top left corner use left and top edge detections
        if tile[1] == 0 and tile[2] == 0:
            collisionLTRBtoUse = [True, True, False, False]
        #if in the top row use top edge detections
        elif tile[1] == 0:
            collisionLTRBtoUse = [False, True, False, False]
        #if in the left colum use left edge detections
        elif tile[2] == 0:
            collisionLTRBtoUse = [True, False, False, False]
        #if in the right column use right edge detections
        elif tile[2] == self.tilledDimensions[0]:
            collisionLTRBtoUse = [False, False, True, False]
        #in in the bottom row use bottom edge detections
        if tile[1] == self.tilledDimensions[1]:
            collisionLTRBtoUse = [False, False, False, True]
        #if in the bottom right corner use bottom and right edge detections
        elif tile[2] == self.tilledDimensions[0] and tile[1] == self.tilledDimensions[1]:
            collisionLTRBtoUse = [False, False, True, True]
        #else use left and top detections
        else:
            collisionLTRBtoUse = [False, False, False, False]

        return collisionLTRBtoUse


    #method to remove overlaping bounding boxes in the overlaping sections of the tiles
    def mMergeOverlaping(self):
        print('merging)')
        pass

 
    #add in functionality to display tile lines
    def mDisplayDetections(self):
        print('display')
        musselCount = 0
        open_musselCount = 0
        for detection in self.detections:
            classId = detection[0]
            score = detection[1]
            box = detection[2]
            if classId == 0:
                musselCount +=1
                cv2.rectangle(self.image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color=(255, 255, 255), thickness=1)
            elif classId == 1:
                open_musselCount +=1
                cv2.rectangle(self.image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color=(0, 255, 0), thickness=1)
            
            
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
        for detection in self.detections:
            if detection[0] == 0:
                cropped_mussel = self.mBBoxCrop(detection[2])
                self.mMeasureContour(cropped_mussel)
    
    #for a given bounding box returns the portion of the image within it
    def mBBoxCrop(self, box):
        #box = [xcord, ycord, boxwidth, boxheight]
        #imagecrop[startheight:endheight, startwidth:endwidth]
        image = self.image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        return image

    #takes in a image of a single detected item and returns the height and width of it
    def mMeasureContour(self, image):
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
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255),-1)
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
                # draw the object sizes on the image
                #cv2.putText(orig, "{:.1f}in".format(distA),(int(tltrX - 10), int(tltrY - 10)), cv2.FONT_HERSHEY_DUPLEX,0.55, (255, 255, 255), 2)
                #cv2.putText(orig, "{:.1f}in".format(distB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_DUPLEX,0.55, (255, 255, 255), 2)
                plt.imshow(orig)
                plt.show()


if __name__ == "__main__":
    app = darkentDetection()
    app.mTileImage()
    app.mDetectMussels()
    app.mMeasureDetectedMussels()
    app.mDisplayDetections()
    

    

