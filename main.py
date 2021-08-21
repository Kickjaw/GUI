import sys, cv2
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox, QFileDialog
from PyQt5.uic import loadUi
from mussel_gui import Ui_MainWindow

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

    def zoomIn(self):
        self.scaleImage(1.25)
    
    def zoomOut(self):
        self.scaleImage(.80)
    
    def scaleImage(self, Factor):
        #get the height of the current window and multiply by the scale factor
        #then set pixmap.scaledtoHeight to that new number
        self.scalingFactor *= Factor
        self.scalingFactor = round(self.pixelMapOriginal.height() * self.scalingFactor)
        self.pixelMapDisplay = self.pixelMapOriginal.scaledToHeight(self.scalingFactor)
        self.updateImage()

    def updateImage(self):
        self.oImageDisplayLabel.setPixmap(self.pixelMapDisplay)


    def openImage(self):
        fname = QFileDialog.getOpenFileName(self)
        self.oLoadedFilePathLabel.setText(fname[0])
        self.oLoadedFilePathLabel.adjustSize()
        self.image = cv2.imread(fname[0])
        self.Qimage = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.pixelMapOriginal = QtGui.QPixmap.fromImage(self.image)
        self.pixelMapDisplay = self.pixelMapOriginal
        self.updateImage()

    #opencv Methods
    #blur = blure factor before applying the circle algortithm
    def circleMask(self, blur, scale):
        modifiedImage = self.image
        modifiedImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        modifiedImage = cv2.medianBlur(modifiedImage, blur)
        
        width = int(modifiedImage.shape[1] * scale / 100)
        height = int(modifiedImage.shape[0] * scale / 100)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())