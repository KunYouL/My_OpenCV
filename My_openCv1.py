import PIL.Image
from PIL import Image, ImageTk
import tkinter as tk
import cv2 as cv
from tkinter import filedialog
from matplotlib import pyplot as plt
import numpy as np
import math


class Myfunction:
    def open_file1(self):
        filetypes = (
            ('png file', '*.png'),
            ('jpg file', '*.jpg'),
            ('All file', '*.*')
        )
        filename = filedialog.askopenfilename(
            title='Open a file',
            initialdir='/',
            filetypes=filetypes
        )
        img = cv.imread(filename)
        return img

    def open_file(self):
        filetypes = (
            ('png file', '*.png'),
            ('jpg file', '*.jpg'),
            ('All file', '*.*')
        )
        filename = filedialog.askopenfilename(
            title='Open a file1',
            initialdir='/',
            filetypes=filetypes
        )
        img = cv.imread(filename)
        cv.imshow('img', img)
        cv.waitKey()

    def save_file(self):
        pass

    def gray(self):  #灰階
        src = self.open_file1()
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        cv.imshow('gray', gray)
        cv.waitKey()

    def HSV(self):  #HSV
        src = self.open_file1()
        HSV = cv.cvtColor(src, cv.COLOR_BGR2HSV)
        cv.imshow('HSV', HSV)
        cv.waitKey()

    def histogram(self):  #直方圖
        scr = self.open_file1()
        cv.imshow('Original image', scr)
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv.calcHist([scr], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.show()

    def hist_equa(self):  # 直方圖等化
        src = self.open_file1()
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        img_eq = cv.equalizeHist(gray)
        cv.imshow('equalized image', img_eq)
        plt.hist(img_eq.ravel(), 256, [0, 256])
        plt.show()

    def canny_detector(self):  #邊緣檢測器
        max_lowThreshold = 100
        window_name = 'Edge Map'
        title_trackbar = 'Min Threshold:'
        ratio = 3
        kernel_size = 3

        #the callback function for trackbar
        def canny_value_change(val):
            low_threshold = val
            img_blur = cv.blur(gray, (3, 3))
            detected_edge = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
            mask = detected_edge != 0
            dst = src * (mask[:, :, None].astype(src.dtype))
            cv.imshow(window_name, dst)

        src = self.open_file1()
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        cv.namedWindow(window_name)
        cv.createTrackbar(title_trackbar, window_name, 0, max_lowThreshold, canny_value_change)
        canny_value_change(0)  #先給數值
        cv.waitKey()

    def hough_transform(self):  #霍夫轉換
        src = cv.cvtColor(self.open_file1(), cv.COLOR_BGR2GRAY)  #先轉灰階
        dst = cv.Canny(src, 50, 200, None, 3)  #零界點，sobel孔大小

        #Copy edges to the image that will display the results in BGR
        cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)   #轉BGR
        cdstP = np.copy(cdst)

        lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)  #1=距離 150=門檻值 0=預設值

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]  #算每條線的ab值
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

        linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255),  3, cv.LINE_AA)

        cv.imshow("Source", src)
        cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)  #標轉霍夫變換
        cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)  #概率線變換

    def gaussianBlur(self): #高斯模糊
        src = self.open_file1()
        Blur = cv.GaussianBlur(src, (15, 15), 10)   #kernel只能是奇數，且逗號前後數值要相同
        cv.imshow('Blur', Blur)

    def corner_harris(self): #邊角檢測
        src = self.open_file1()
        gray= cv.cvtColor(src,cv.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv.cornerHarris(gray,2,3,0.06)
        dst = cv.dilate(dst,None)
        src[dst>0.01*dst.max()] = [0,0,255]
        cv.imshow('src',src)
        cv.imshow('dst',dst)

    def thresholding(self):  #二值化
        src = self.open_file1()
        src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

        ret, th1 = cv.threshold(src, 127, 255, cv.THRESH_BINARY)  #二值化(未模糊)
        #adaptiveThreshold可將一般圖片做自適應二值化
        th2 = cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2) #  平均二值化(未模糊降噪)
        th3 = cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2) #高斯二值化(未模糊ru)

        cv.imshow('th1', th1)
        cv.imshow('th2', th2)
        cv.imshow('th3', th3)

    def transpose(self):  #水平翻轉
        src = self.open_file1()
        cv.imshow('src', src)
        new_src = cv.flip(src, 1)
        cv.imshow('new_src', new_src)

    def warpperspective(self):  #影像透視
        src = self.open_file1()
        cv.circle(src, (18, 28), 5, (255, 128, 128), -1)  # (圖片, 圓的位子, 圓大小,顏色, 實心 )
        cv.circle(src, (202, 32), 5, (255, 128, 128), -1)
        cv.circle(src, (17, 161), 5, (255, 128, 128), -1)
        cv.circle(src, (186, 168), 5, (255, 128, 128), -1)

        pts1 = np.float32([[18, 28], [202, 32], [17, 150], [186, 150]])  # 第二圖對應的點
        pts2 = np.float32([[0, 0], [400, 0], [0, 600], [400, 500]])  # 第二圖的大小
        matrix = cv.getPerspectiveTransform(pts1, pts2)  # 轉換

        result = cv.warpPerspective(src, matrix, (400, 500))
        cv.imshow('src',src)
        cv.imshow('result',result)
        pass

    def affine(self):  #仿射轉換(平移)
        src = self.open_file1()
        height, width = src.shape[:2]  #讀取原影像的長、寬
        x = 30  #自訂義轉換矩陣M的x軸移動值
        y = 20  #自訂義轉換矩陣M的y軸移動值
        M = np.float32([[1, 0, x], [0, 1, y]])
        move = cv.warpAffine(src, M, (width, height))   #平移映射
        cv.imshow('src', src)
        cv.imshow('move', move)

    def rotate(self):  #仿射轉換(旋轉)
        src = self.open_file1()
        height, width = src.shape[:2] #讀取原影像的長、寬
        M=cv.getRotationMatrix2D((width/2,height/2),45,0.6)  #以中心為原點，逆時針轉45度，且縮小為原圖的0.6倍
        rotate=cv.warpAffine(src,M,(width,height))   #旋轉映射
        cv.imshow('src', src)
        cv.imshow('rotate', rotate)