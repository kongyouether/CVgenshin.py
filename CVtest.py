# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CVtest.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


import sys, cv2, time, threading  # 导入系统模块,OpenCV模块
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton,QDesktopWidget # 导入PyQt5模块
from PyQt5.QtGui import QIcon, QPixmap
import numpy as np
import random


class Ui_CVtest(QWidget):
    def __init__(self):  # * 初始化函数
        super().__init__()
        screen = QDesktopWidget().screenGeometry()
        self.move(int((screen.width() - 900) / 2), int((screen.height() - 1000) / 2))
        self.background()

        self.setupUi(self)


    def setupUi(self, CVtest):
        CVtest.setObjectName("CVtest")
        CVtest.resize(907, 801)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        CVtest.setFont(font)
        self.gridLayout = QtWidgets.QGridLayout(CVtest)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(CVtest)
        self.label.setEnabled(True)
        self.label.setMinimumSize(QtCore.QSize(0, 30))
        self.label.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.horizontalSlider = QtWidgets.QSlider(CVtest)
        self.horizontalSlider.setMinimumSize(QtCore.QSize(0, 30))
        self.horizontalSlider.setMaximumSize(QtCore.QSize(16777215, 30))
        self.horizontalSlider.setMaximum(10)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalSlider.valueChanged.connect(self.valueChange)

        self.gridLayout.addWidget(self.horizontalSlider, 0, 1, 1, 1)
        self.label_number = QtWidgets.QLabel(CVtest)
        font = QtGui.QFont()
        font.setFamily("得意黑")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_number.setFont(font)
        self.label_number.setAlignment(QtCore.Qt.AlignCenter)
        self.label_number.setObjectName("label_number")
        self.gridLayout.addWidget(self.label_number, 0, 2, 1, 1)
        self.pushButton = QtWidgets.QPushButton(CVtest)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.generate_point)
        self.gridLayout.addWidget(self.pushButton, 0, 3, 1, 1)
        self.line = QtWidgets.QFrame(CVtest)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 1, 0, 1, 4)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(57, 17, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.graphicsView = QtWidgets.QGraphicsView(CVtest)
        self.graphicsView.setEnabled(True)
        self.graphicsView.setMinimumSize(QtCore.QSize(800, 800))
        self.graphicsView.setMaximumSize(QtCore.QSize(800, 800))
        self.graphicsView.setMouseTracking(True)
        self.graphicsView.setAlignment(QtCore.Qt.AlignCenter)
        self.graphicsView.setObjectName("graphicsView")
        # graphicsView展示图片white.png
        self.scene = QtWidgets.QGraphicsScene()
        self.scene.addPixmap(QtGui.QPixmap("white.png"))
        self.graphicsView.setScene(self.scene)

        self.horizontalLayout.addWidget(self.graphicsView)
        spacerItem1 = QtWidgets.QSpacerItem(68, 697, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.gridLayout.addLayout(self.horizontalLayout, 2, 0, 1, 4)
        self.label.setBuddy(self.horizontalSlider)

        self.retranslateUi(CVtest)
        QtCore.QMetaObject.connectSlotsByName(CVtest)
        self.show()

    def retranslateUi(self, CVtest):
        _translate = QtCore.QCoreApplication.translate
        CVtest.setWindowTitle(_translate("CVtest", "CVtest生成器"))
        CVtest.setWindowIcon(QIcon("./cam.ico"))  # 设置窗口图标
        self.label.setText(_translate("CVtest", "圆点数量："))
        self.label_number.setText(_translate("CVtest", "0"))
        self.pushButton.setText(_translate("CVtest", "生成"))

    def valueChange(self):
        global number
        number = self.horizontalSlider.value()
        # print("number:", number)
        self.label_number.setText(str(number))

    def background(self):
        global img
        img = np.zeros((1400, 1400, 3), np.uint8)
        img.fill(255)
        # 在图片的四角生成定位点
        img[100:200, 100:200] = [0, 0, 0]
        img[100:200, 1200:1300] = [0, 0, 0]
        img[1200:1300, 100:200] = [0, 0, 0]
        # img[1200:1300, 1200:1300] = [0, 0, 0]

        img[115:185, 115:185] = [255, 255, 255]
        img[115:185, 1215:1285] = [255, 255, 255]
        img[1215:1285, 115:185] = [255, 255, 255]
        # img[1215:1285, 1215:1285] = [255, 255, 255]

        img[130:170, 130:170] = [0, 0, 0]
        img[130:170, 1230:1270] = [0, 0, 0]
        img[1230:1270, 130:170] = [0, 0, 0]
        # img[1230:1270, 1230:1270] = [0, 0, 0]

        # 在图片中心生成一个800*800的黑色区域
        img[290:1110, 290:1110] = [0, 0, 0]

        # 在图片中生成64个白色方块，每个方块大小为80*80，间隔为20
        for i in range(8):
            for j in range(8):
                img[310 + i * 100: 390 + i * 100, 310 + j * 100: 390 + j * 100] = [255, 255, 255]

        # 在图片中生成1-8，a-h的标识
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(9):
            # 如果i = 8，就分别写x和y
            if i == 8:
                cv2.putText(img, 'x', (310 + i * 100 + 30, 280), font, 2, (180, 0, 0), 6)
                cv2.putText(img, 'y', (260, 330 + i * 100 + 30), font, 2, (0, 0, 180), 6)
            else:
                cv2.putText(img, str(i + 1), (310 + i * 100 + 30, 280), font, 1, (0, 0, 0), 2)
                cv2.putText(img, str(i + 1), (260, 330 + i * 100 + 30), font, 1, (0, 0, 0), 2)
        img = cv2.resize(img, (700, 700), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite("white.png", img)

    def generate_point(self):
        global img, number
        gene_img = img.copy()
        # 在图片中生成number个随机圆点
        # 生成数组记录已经生成的圆点的位置
        # 生成随机数，如果随机数已经在数组中，就重新生成
        # 如果随机数不在数组中，就将随机数加入数组
        point_list = []
        for i in range(number):
            while True:
                x = random.randint(0, 7)
                y = random.randint(0, 7)
                if [x, y] not in point_list:
                    point_list.append([x, y])
                    cv2.circle(gene_img, (175 + 50 * x, 175 + 50 * y), 12, (0, 0, 0), -1)
                    break

        # print("x = ", xx + 1, "y = ", yy + 1)
        # cv2.circle(img, (350 + 100 * xx, 350 + 100 * yy), 25, (0, 0, 0), -1)
        # cv2.imshow('image', img)
        cv2.imwrite('test.png', gene_img)
        # 将生成的图片展示在graphicsView中
        self.scene = QtWidgets.QGraphicsScene()
        self.scene.addPixmap(QtGui.QPixmap("test.png"))
        self.graphicsView.setScene(self.scene)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Ui_CVtest()
    sys.exit(app.exec_())