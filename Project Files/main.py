import sys, cv2
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox, QFileDialog
from PyQt5.uic import loadUi
from mussel_gui import Ui_MainWindow
from yolov5_detection import yolov5Detection
import numpy as np
from PIL import Image
from logger import logger

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.detection = yolov5Detection()
        self.scalingFactor = 1
        self.pixelMapDisplay = QtGui.QPixmap(1,1)
        self.pixelMapOriginal = QtGui.QPixmap(1,1)
        self.setupUi(self)
        self.logger = logger()
        self.connectSignalsSlots()

    def connectSignalsSlots(self):
        self.actionOpen.triggered.connect(self.openImage)
        self.actionZoom_In.triggered.connect(self.zoomIn)
        self.actionZoom_Out.triggered.connect(self.zoomOut)
        self.action100.triggered.connect(self.set100)
        self.action75.triggered.connect(self.set75)
        self.action50.triggered.connect(self.set50)
        self.action25.triggered.connect(self.set25)
        self.oDetectionButton.clicked.connect(self.detectionButton)

    def zoomIn(self):
        self.scaleImage(1.25)
    def zoomOut(self):
        self.scaleImage(.80)
    def set100(self):
        self.scaleImage(1)
    def set75(self):
        self.scaleImage(.75)
    def set50(self):
        self.scaleImage(.5)
    def set25(self):
        self.scaleImage(.25)

    
    def detectionButton(self):
        self.logger.recordLog('Status', 'Detection button hit starting detection with confidence level' + self.detection.confidenceCutOff)
        self.detection.confidenceCutOff = int(self.oConfidenceLevelInput.text())/100
        self.detection.image = Image.open(self.oLoadedFilePathLabel.text())
        self.detection.imagecv2 = self.image
        self.detection.mDetection()
        self.updateDetectionDisplay()
        
        
    
    def scaleImage(self, Factor):
        #get the height of the current window and multiply by the scale factor
        #then set pixmap.scaledtoHeight to that new number
        self.scalingFactor *= Factor
        scaledHeight = round(self.pixelMapOriginal.height() * self.scalingFactor)
        self.pixelMapDisplay = self.pixelMapOriginal.scaledToHeight(scaledHeight)
        self.updateImage()

    def updateImage(self):
        self.oImageDisplayLabel.setPixmap(self.pixelMapDisplay)


    def openImage(self):
        fname = QFileDialog.getOpenFileName(self)
        self.oLoadedFilePathLabel.setText(fname[0])
        self.oLoadedFilePathLabel.adjustSize()
        self.image = cv2.imread(fname[0])
        self.Qimage = self.QImageFromCV2Image(self.image)
        self.pixelMapOriginal = QtGui.QPixmap.fromImage(self.Qimage)
        self.pixelMapDisplay = self.pixelMapOriginal
        #auto scale the image to the window size on load
        scaleFactor = self.oImageDisplayLabel.height() / self.image.shape[0]
        self.scaleImage(scaleFactor)


    def updateDetectionDisplay(self):
        self.scalingFactor = 1
        self.detection.mDisplayDetections()
        self.Qimage = self.QImageFromCV2Image(self.detection.imagecv2)
        
        self.pixelMapOriginal = QtGui.QPixmap.fromImage(self.Qimage)
        self.pixelMapDisplay = self.pixelMapOriginal

        #auto scale the image to the window size on load
        scaleFactor = self.oImageDisplayLabel.height() / self.image.shape[0]
        self.scaleImage(scaleFactor)
        self.oMusselCountLabel.setText(str(self.detection.musselCount))
        self.oOpenMusselCountLabel.setText(str(self.detection.openMusselCount))
        self.oFullOpenMusselCountLabel.setText(str(self.detection.fullOpenMusselCount))

    
    def QImageFromCV2Image(self, cv2image):
        # get the shape of the array
        height, width, depth = np.shape(cv2image)

        # calculate the total number of bytes in the frame 
        totalBytes = cv2image.nbytes

        # divide by the number of rows
        bytesPerLine = int(totalBytes/height)

        Qimage = QtGui.QImage(cv2image, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()

        return Qimage



    




if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()

    

    sys.exit(app.exec())