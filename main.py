import sys, cv2
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox, QFileDialog
from PyQt5.uic import loadUi
from mussel_gui import Ui_MainWindow
import numpy as np

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__()

        self.scalingFactor = 1
        self.pixelMapDisplay = QtGui.QPixmap(1,1)
        self.pixelMapOriginal = QtGui.QPixmap(1,1)
        self.setupUi(self)
        self.connectSignalsSlots()

    def connectSignalsSlots(self):
        self.actionOpen.triggered.connect(self.openImage)
        self.actionZoom_In.triggered.connect(self.zoomIn)
        self.actionZoom_Out.triggered.connect(self.zoomOut)
        self.action100.triggered.connect(self.set100)
        self.action75.triggered.connect(self.set75)
        self.action50.triggered.connect(self.set50)
        self.action25.triggered.connect(self.set25)
        self.oCircleMaskButton.clicked.connect(self.circleMaskButton)

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
    def circleMaskButton(self):
        self.circleMask(5, .1)
    
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
        self.Qimage = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.pixelMapOriginal = QtGui.QPixmap.fromImage(self.Qimage)
        self.pixelMapDisplay = self.pixelMapOriginal

        #auto scale the image to the window size on load
        scaleFactor = self.oImageDisplayLabel.height() / self.image.shape[0]
        self.scaleImage(scaleFactor)






if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())