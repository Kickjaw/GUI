import cv2
import matplotlib.pyplot as plt
import numpy as np
import time as time




#methods to tile input image, detect mussels, reassemble image with detections

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
        plt.imshow(im2)
        plt.show()
    


if __name__ == "__main__":
    app = darkentDetection()
    app.mTileImage()
    app.mDetectMussels()
    app.mDisplayDetections()
    

    

