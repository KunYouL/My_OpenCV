import tkinter as tk
import PIL.Image
import numpy as np
from PIL import ImageTk, Image
import cv2 as cv
from tkinter import filedialog
from matplotlib import pyplot as plt
import My_openCv1

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title('My window')
        self.window.geometry('1000x600')

        self.my_function = My_openCv1.Myfunction()

        #Add a menubar
        self.main_menu = tk.Menu(window)  #創建視窗

        #Add file submenu
        self.file_menu = tk.Menu(self.main_menu, tearoff=0)
        self.file_menu.add_command(label="開啟檔案", command=self.my_function.open_file)
        self.file_menu.add_command(label="儲存檔案", command=self.my_function.save_file)
        self.file_menu.add_separator()  #分隔線
        self.file_menu.add_command(label="離開程式", command=window.quit)

        #Add operation1 submenu
        self.operation1_menu = tk.Menu(self.main_menu, tearoff=0)
        self.operation1_menu.add_command(label='邊緣檢測器', command=self.my_function.canny_detector)
        self.operation1_menu.add_command(label='霍夫轉換', command=self.my_function.hough_transform)
        self.operation1_menu.add_command(label='高斯模糊', command=self.my_function.gaussianBlur)
        self.operation1_menu.add_command(label='邊角偵測', command=self.my_function.corner_harris)
        self.operation1_menu.add_command(label='輪廓檢測', command=self.my_function.contour)
        self.operation1_menu.add_command(label='輪廓檢測(1)', command=self.my_function.find_contour)

        #Add operation2 submenu
        self.operation2_menu = tk.Menu(self.main_menu, tearoff=0)
        self.operation2_menu.add_command(label='灰階圖', command=self.my_function.gray)
        self.operation2_menu.add_command(label='HSV', command=self.my_function.HSV)
        self.operation2_menu.add_command(label='水平翻轉', command=self.my_function.transpose)

        # Add operation3 submenu
        self.operation3_menu = tk.Menu(self.main_menu, tearoff=0)
        self.operation3_menu.add_command(label='直方圖', command=self.my_function.histogram)
        self.operation3_menu.add_command(label='直方圖等化', command=self.my_function.hist_equa)
        self.operation3_menu.add_command(label='二值化', command=self.my_function.thresholding)

        # Add operation4 submenu
        self.operation4_menu = tk.Menu(self.main_menu, tearoff=0)
        self.operation4_menu.add_command(label='影像透視', command=self.my_function.warpperspective)
        self.operation4_menu.add_command(label='仿射轉換(平移)', command=self.my_function.affine)
        self.operation4_menu.add_command(label='仿射轉換(旋轉)', command=self.my_function.rotate)

        #Add  submenu to mainmenu
        self.main_menu.add_cascade(label="檔案", menu=self.file_menu)
        self.main_menu.add_cascade(label='功能1', menu=self.operation1_menu)
        self.main_menu.add_cascade(label='功能2', menu=self.operation2_menu)
        self.main_menu.add_cascade(label='功能3', menu=self.operation3_menu)
        self.main_menu.add_cascade(label='功能4', menu=self.operation4_menu)

        #display menu
        self.window.config(menu=self.main_menu)
        self.window.mainloop()


App(tk.Tk(), "OpenCv with Tkinter GUI")

