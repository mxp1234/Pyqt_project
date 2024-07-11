import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy
import dataProcess
from dataProcess import *

if __name__ =='__main__':
    application = QApplication(sys.argv)
    # aw = QMainWindow()
    w = dataProcess()
    #Ui_MainWindow()里没有show函数，show函数是位于QMainWindow里
    w.show()
    #之所以还要实例化一个QMainWindow()，因为生成的Ui_MainWindow()类的setupUi里还要传入一个外来参数。
    
    
    sys.exit(application.exec_())