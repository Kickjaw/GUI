import cv2
import matplotlib.pyplot as plt
import numpy as np
import time as time




#methods to tile input image, detect mussels, reassemble image with detections

class darkentDetection(object):

    def __init__(self):
        self.image = cv2.imread('3.jpg')
        self.tilesize = 512
        self.overlap = 40
        self.classes = ['mussel', 'mussel_open']
        self.net = cv2.dnn.readNetFromDarknet('yolov4-obj-tilled.cfg', 'yolov4-obj-tilled_best.weights')
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(scale=1 / 255, size=(512, 512), swapRB=True)
        self.tiles = [] #list of tiles, tile = [tilledImage, tileYcord, tileXcord]
        self.detections = [] #list of detected items, = []


    #takes in an image and overlap and returns a list of images
    #returnes list of images all the same size, if the cut tile is to small pad with black pixels
    def mTileImage(self):
        x_numCopies = int(np.ceil(self.image.shape[1]/(self.tilesize-(self.overlap*2))))-1
        y_numCopies = int(np.ceil(self.image.shape[0]/(self.tilesize-(self.overlap*2))))-1

        for i in range(y_numCopies):
            for j in range(x_numCopies):
                #imagecrop[startheight:endheight, startwidth:endwidth]
                #if the top left corner
                if j == 0 and i == 0:
                    tempImage = self.image[0:(self.tilesize)*(i+1),0:(self.tilesize)*(j+1)]
                #if left most image
                elif j ==0:
                    tempImage = self.image[(self.tilesize)*i-self.overlap:(self.tilesize)*(i+1)-self.overlap,0:(self.tilesize)*(j+1)]
                #if in top row
                elif i ==0:
                    tempImage = self.image[0:(self.tilesize)*(i+1),(self.tilesize)*j-self.overlap:(self.tilesize)*(j+1)-self.overlap]
                else:
                    tempImage = self.image[(self.tilesize)*i-self.overlap:(self.tilesize)*(i+1)-self.overlap,(self.tilesize)*j-self.overlap:(self.tilesize)*(j+1)-self.overlap]


                #pad the image with blackness if it not large enough to be tile sized
                if tempImage.shape[1] != self.tilesize:
                    tempImage = cv2.copyMakeBorder(tempImage, 0, 0, 0, (self.tilesize-tempImage.shape[1]),cv2.BORDER_CONSTANT,value=[0,0,0])
                if tempImage.shape[0] != self.tilesize:
                    tempImage = cv2.copyMakeBorder(tempImage, 0, (self.tilesize-tempImage.shape[0]), 0, 0, cv2.BORDER_CONSTANT,value=[0,0,0])
                self.tiles.append([tempImage, i, j])

    #takes all the tiles and returns all the bounding boxes
    def mDetectMussels(self):
        #run the detection
        for tile in self.tiles:
            classIds, scores, boxes = self.model.detect(tile[0], confThreshold=0.6, nmsThreshold=0.4)
            #box = [xcord, ycord, boxwidth, boxheight]
            #add in for loop to go over boxes and append them to the main box list, while translating them in
            #space so that they line up with the original
            for (classId, score, box) in zip(classIds, scores, boxes):
                if tile[1] == 0 and tile[2] == 0:
                    boxTranslated = box
                elif tile[2] == 0:
                    boxTranslated = [box[0], box[1]+(tile[1]*512-self.overlap), box[2], box[3]]
                elif tile[1] == 0:
                    boxTranslated = [box[0]+(tile[2]*512-self.overlap), box[1], box[2], box[3]]
                else:
                    boxTranslated = [box[0]+(tile[2]*512-self.overlap), box[1]+(tile[1]*512-self.overlap), box[2], box[3]]
                self.detections.append([classId, score, boxTranslated])

            #     cv2.rectangle(tile[0], (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
            #                 color=(0, 255, 0), thickness=2)
            
            #     text = '%s: %.2f' % (self.classes[classId[0]], score)
            #     cv2.putText(tile[0], text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #                 color=(0, 255, 0), thickness=2)
            
            # cv2.imshow('Image', tile[0])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    def mDisplayDetections(self):
        for detection in self.detections:
            classId = detection[0]
            score = detection[1]
            box = detection[2]
            cv2.rectangle(self.image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                            color=(0, 255, 0), thickness=2)
            
            text = '%s: %.2f' % (self.classes[classId[0]], score)
            cv2.putText(self.image, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color=(0, 255, 0), thickness=2) 
        plt.imshow(self.image)
        plt.show()
    


if __name__ == "__main__":
    app = darkentDetection()
    app.mTileImage()
    app.mDetectMussels()
    app.mDisplayDetections()
    

    

