import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QGraphicsView
from PyQt5.QtGui import QFont
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QGraphicsView
from PyQt5 import QtWidgets
import sys
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton, QLabel, QFileDialog, QSplitter, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from cv_test import test_main
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
import cv2
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QIcon
import os
global image_chuan
import subprocess
global result_image
global image_zhongchuan


class MyMainWindow(QMainWindow):
    def __init__(self):
        global image_chuan
        super().__init__()
        # 设置窗口图标
        icon = QIcon("xiaofengmian.ico")  # 将 "your_icon.ico" 替换为你的ICO文件路径
        self.setWindowIcon(icon)
        # 创建UI对象并初始化
        self.setupUi(self)
        self.action_3.triggered.connect(self.open_dev_log)
        # 在这里可以连接按钮的点击事件
        self.pushButton.clicked.connect(self.select_image)
        self.pushButton_2.clicked.connect(self.process_image)
        self.pushButton_3.clicked.connect(self.refresh)
        self.pushButton_4.clicked.connect(self.show_civilization)

    def setupUi(self, MainWindow):
        global image_chuan
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1300, 960)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(1300, 960))
        MainWindow.setMaximumSize(QtCore.QSize(1300, 960))
        MainWindow.setAutoFillBackground(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(70, 820, 1161, 41))
        font = QtGui.QFont()
        font.setFamily("微软雅黑 Light")
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(70, 50, 217, 122))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.pushButton = QtWidgets.QPushButton(self.widget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑 Light")
        font.setPointSize(14)
        self.pushButton.setFont(font)
        self.pushButton.setCheckable(False)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_3.addWidget(self.pushButton)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem)
        self.pushButton_2 = QtWidgets.QPushButton(self.widget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑 Light")
        font.setPointSize(14)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setCheckable(False)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout_3.addWidget(self.pushButton_2)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.pushButton_3 = QtWidgets.QPushButton(self.widget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑 Light")
        font.setPointSize(14)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setCheckable(False)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout_4.addWidget(self.pushButton_3)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem2)
        self.pushButton_4 = QtWidgets.QPushButton(self.widget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑 Light")
        font.setPointSize(14)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setCheckable(False)
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout_4.addWidget(self.pushButton_4)
        self.horizontalLayout.addLayout(self.verticalLayout_4)
        self.widget1 = QtWidgets.QWidget(self.centralwidget)
        self.widget1.setGeometry(QtCore.QRect(20, 200, 1252, 584))
        self.widget1.setObjectName("widget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget1)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.graphicsView_3 = QtWidgets.QGraphicsView(self.widget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graphicsView_3.sizePolicy().hasHeightForWidth())
        self.graphicsView_3.setSizePolicy(sizePolicy)
        self.graphicsView_3.setMinimumSize(QtCore.QSize(550, 550))
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.verticalLayout_2.addWidget(self.graphicsView_3)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem4)
        self.label_3 = QtWidgets.QLabel(self.widget1)
        font = QtGui.QFont()
        font.setFamily("微软雅黑 Light")
        font.setPointSize(11)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_5.addWidget(self.label_3)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem5)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6.addLayout(self.verticalLayout_2)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem6)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.widget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graphicsView_2.sizePolicy().hasHeightForWidth())
        self.graphicsView_2.setSizePolicy(sizePolicy)
        self.graphicsView_2.setMinimumSize(QtCore.QSize(550, 550))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.verticalLayout.addWidget(self.graphicsView_2)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem7)
        self.label_2 = QtWidgets.QLabel(self.widget1)
        font = QtGui.QFont()
        font.setFamily("微软雅黑 Light")
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_4.addWidget(self.label_2)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem8)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_6.addLayout(self.verticalLayout)
        self.horizontalLayout_2.addLayout(self.horizontalLayout_6)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem9)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1300, 22))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionV1_0 = QtWidgets.QAction(MainWindow)
        self.actionV1_0.setObjectName("actionV1_0")
        self.action_3 = QtWidgets.QAction(MainWindow)
        self.action_3.setObjectName("action_3")
        self.menu.addAction(self.actionV1_0)
        self.menu.addSeparator()
        self.menu_2.addSeparator()
        self.menu_2.addAction(self.action_3)
        self.menu_2.addSeparator()
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())


        self.retranslateUi(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "原神"))
        self.label.setText(_translate("MainWindow", "所有圆的坐标："))
        self.pushButton.setText(_translate("MainWindow", "选择图片"))
        self.pushButton_2.setText(_translate("MainWindow", "处理图片"))
        self.pushButton_3.setText(_translate("MainWindow", "刷新"))
        self.pushButton_4.setText(_translate("MainWindow", "启动"))
        self.label_3.setText(_translate("MainWindow", "原图片"))
        self.label_2.setText(_translate("MainWindow", "处理后图片"))
        self.menu.setTitle(_translate("MainWindow", "系统"))
        self.menu_2.setTitle(_translate("MainWindow", "关于"))
        self.actionV1_0.setText(_translate("MainWindow", "版本V1.0"))
        self.action_3.setText(_translate("MainWindow", "开发日志"))

    def open_dev_log(self):
        # 打开记事本或文本编辑器以查看开发日志
        dev_log_path = "Development_Log.txt"  # 将此路径替换为实际的开发日志文件路径
        if os.path.exists(dev_log_path):
            os.system(f"notepad.exe {dev_log_path}")
        else:
            print("开发日志文件不存在！")

    def select_image(self):
        global image_chuan
        global image_zhongchuan
        # 打开文件对话框以选择图片文件
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly  # 只读模式
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.bmp *.gif *.jpeg)", options=options)
        image_chuan = cv2.imread(file_name)
        #cv2.imshow("image_chuan1",image_chuan)
        if image_chuan is not None:
            image_zhongchuan=image_chuan.copy()
        #cv2.imshow("image_zhongchuan",image_zhongchuan)

        # 如果用户选择了图片文件，则显示在标签上，并设置标题文本
        if file_name:
            # 加载图片
            image = QImage(file_name)

            # 计算缩放比例以确保像素面积不超过700000
            max_area = 300000
            current_area = image.width() * image.height()
            scale_factor = 1.0

            while current_area > max_area or (max_area - current_area) > 1000:
                scale_factor = (max_area / current_area) ** 0.5  # 使用0.5作为指数来等比缩放
                new_size = QSize(int(image.width() * scale_factor), int(image.height() * scale_factor))
                image = image.scaled(new_size, Qt.KeepAspectRatio)
                current_area = image.width() * image.height()

            pixmap = QPixmap.fromImage(image)

            # 设置原始图片标签的 pixmap
            #self.original_image_label.setPixmap(pixmap)

            # 创建 QGraphicsScene
            scene = QGraphicsScene()
            # 创建 QGraphicsPixmapItem 并添加到场景中
            pixmap_item = QGraphicsPixmapItem(pixmap)
            scene.addItem(pixmap_item)

            # 将场景设置给 self.graphicsView_2
            self.graphicsView_3.setScene(scene)
        


    def process_image(self):
        global image_zhongchuan
        global processed_image
        global all_circle_coordinates
        global result_image
        print("进入 process_image 函数") 
        global image_chuan
        # 获取当前显示的原始图片
        scene = self.graphicsView_3.scene()

        if scene:
            print("进入 if scene:") 
            # 获取原始图片的 QGraphicsPixmapItem
            pixmap_item = scene.items()[0] if scene.items() else None

            if pixmap_item:
                print("进入if pixmap_item:") 
                # 获取原始图片的 QPixmap
                pixmap = pixmap_item.pixmap()


                print("进入 if pixmap:")    
                # 将 QPixmap 转换为 QImage
                #image = pixmap.toImage()

                # 创建新的 QPixmap 并设置到处理后的图片标签
                gray_pixmap, circles,result_image = test_main(image_zhongchuan)
                image_zhongchuan=image_chuan.copy()
                #调试用
                #cv2.imshow("image_chuan",image_chuan)
                #cv2.imshow("image_zhongchuan",image_zhongchuan)

                # 计算缩放比例以确保像素面积不超过150000
                max_area = 300000
                current_area = gray_pixmap.width() * gray_pixmap.height()
                scale_factor = 1.0
                while current_area > max_area or (max_area - current_area) > 1000:
                    scale_factor = (max_area / current_area) ** 0.5  # 使用0.5作为指数来等比缩放
                    new_size = QSize(int(gray_pixmap.width() * scale_factor), int(gray_pixmap.height() * scale_factor))
                    gray_pixmap = gray_pixmap.scaled(new_size, Qt.KeepAspectRatio)
                    current_area = gray_pixmap.width() * gray_pixmap.height()

                # 将 QImage 转换为 QPixmap
                gray_pixmap = QPixmap.fromImage(gray_pixmap)

                # 创建新的 QGraphicsPixmapItem 并设置到处理后的图片标签
                pixmap_item_processed = QGraphicsPixmapItem(gray_pixmap)
                scene_processed = QGraphicsScene()
                scene_processed.addItem(pixmap_item_processed)
                
                # 设置处理后的图片标签的场景
                self.graphicsView_2.setScene(scene_processed)

                # 显示所有圆的坐标
                circle_text = "所有圆的坐标：{}".format(circles)
                # 添加到现有的文本后面
                # 直接设置标签的文本
                self.label.setText("" + circle_text)
                # 在这里将结果赋值给全局变量 result_image
                #result_image = gray_pixmap.toImage()

    def refresh(self):
        # 清除显示的图片
        self.graphicsView_2.setScene(None)

        self.graphicsView_3.setScene(None)

        # 清除显示的圆坐标
        self.label.setText("所有圆的坐标：")
    def show_civilization(self):
        # 获取 "wenming.jpg" 图片的路径
        image_path = "../wenming.jpg"

        if os.path.exists(image_path):
            try:
                # 使用默认的图像查看器打开图片
                subprocess.Popen(['start', ' ', image_path], shell=True)
            except Exception as e:
                print(f"无法打开图片: {str(e)}")
        else:
            print("图片文件不存在！")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
