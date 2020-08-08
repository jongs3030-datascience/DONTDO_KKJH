#/main
import cv2
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
'''
#from gui_viewer import GUIViewer#
from render_thread import RenderThread
if __name__ == '__main__':
    #app = QtWidgets.QApplication(sys.argv)

    #viewer = GUIViewer()
    #viewer.show()
    
    #app.exec_()
    app=RenderThread()
    app.run()
