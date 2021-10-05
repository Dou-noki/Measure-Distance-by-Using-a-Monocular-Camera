#-*- coding: utf-8 -*-
#编写人：邓轩
#开发时间：2021/9/12 12:57

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import windnd
import matplotlib.pyplot as plt
import cv2 as cv
import time
import numpy as np
import random

import torch


from init_model import init_model

init_model = init_model()
init_model()

class Tkwindow:
    """
    窗口类，用于生成主窗口
    """
    def __init__(self, root):

        self.root = root  # 父窗口
        self.width = 640  # 帆布组件的宽度
        self.height = 480  # 帆布组件的高度

        # self.work = tk.Frame(root)
        self.button1 = tk.Button(self.root, width=10, height=2, bg='white', text='原始摄像头',command=self.click_button1)
        self.button2 = tk.Button(self.root, width=10, height=2, bg='white', text='深度摄像头', command=self.click_button2)
        self.button3 = tk.Button(self.root, width=10, height=2, bg='white', text='物体识别', command=self.click_button3)
        self.button4 = tk.Button(self.root, width=10, height=2, bg='white', text='人体测距', command=self.click_button4)
        self.button5 = tk.Button(self.root, width=10, height=2, bg='white', text='驾驶模式', command=self.click_button5)
        self.work_output_cv = tk.Canvas(self.root, width=self.width, height=self.height, bg='white')  # 帆布

    def init_cv(self):
        """
        初始化摄像头显示区域
        """
        self.work_output_cv.text = self.work_output_cv.create_text(self.width / 2, self.height / 2,
                                                                             text='请点击左侧按钮选择对应功能', fill='grey', anchor='center')
        self.work_output_cv.pack(side='left')

    def init_button(self):
        '''
        初始化四个按钮
        '''
        self.button1.place(x=780, y=int(480 / 6)*1-40, anchor='nw')
        self.button2.place(x=780, y=int(480 / 6)*2-30, anchor='nw')
        self.button3.place(x=780, y=int(480 / 6)*3-20, anchor='nw')
        self.button4.place(x=780, y=int(480 / 6)*4-10, anchor='nw')
        self.button5.place(x=780, y=int(480 / 6)*5, anchor='nw')

    def click_button1(self):
        cap = cv.VideoCapture(0)
        while (True):
            ref, frame = cap.read()
            frame = cv.flip(frame, 1)  # 摄像头翻转
            cvimage = cv.cvtColor(frame,cv.COLOR_BGR2RGBA)
            pilImage = Image.fromarray(cvimage)
            tkImage = ImageTk.PhotoImage(image=pilImage)
            self.work_output_cv.create_image(0, 0, anchor='nw', image=tkImage)
            self.root.update()
            self.root.after(10)
        return



    def click_button2(self):
        cap = cv.VideoCapture(0)
        while (True):
            ref, frame = cap.read()
            frame = cv.flip(frame, 1)  # 摄像头翻转
            img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            input_batch = init_model.transform(img).cuda()


            with torch.no_grad():
                prediction = init_model.model(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

                output = prediction.cpu().numpy()
                output = ((output-output.min())/(output.max()-output.min()))*255
                output_int = output.astype(np.uint8)
                output = cv.cvtColor(output_int,cv.COLOR_GRAY2BGR)
                output = cv.applyColorMap(output,cv.COLORMAP_TWILIGHT_SHIFTED)

                output = cv.cvtColor(output,cv.COLOR_BGR2RGBA)
                pilImage = Image.fromarray(output)
                tkImage = ImageTk.PhotoImage(image=pilImage)
            self.work_output_cv.create_image(0, 0, anchor='nw', image=tkImage)
            self.root.update()
            self.root.after(10)

        cv.destroyAllWindows()
        cap.release()
        return




    def click_button3(self):
        cap = cv.VideoCapture(0)

        # 读取参考
        ref_person = cv.imread('ReferenceImages/1.jpg')

        person_data = init_model.object_detector(ref_person)
        person_width_in_rf = person_data[0][1]  # 得到人的像素宽度

        # 寻找焦距
        focal_person = init_model.focal_length_finder(init_model.KNOWN_DISTANCE, init_model.PERSON_WIDTH, person_width_in_rf)

        while (True):
            ref, frame = cap.read()
            frame = cv.flip(frame, 1)  # 摄像头翻转
            img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            classes, scores, boxes = init_model.model2.detect(frame, init_model.CONFIDENCE_THRESHOLD, init_model.NMS_THRESHOLD)

            for (classid, score, box) in zip(classes, scores, boxes):
                color = init_model.COLORS[int(classid) % len(init_model.COLORS)]
                label = "%s" % (init_model.class_names[classid[0]])

                # 在对象上绘制矩形并在对象上添加标签
                cv.rectangle(frame, box, color, 2)
                frame = init_model.cv2_image_add_Chinese(frame, label, box[0] + 5, box[1] - 28, color, 20)
            output = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
            pilImage = Image.fromarray(output)
            tkImage = ImageTk.PhotoImage(image=pilImage)
            self.work_output_cv.create_image(0, 0, anchor='nw', image=tkImage)
            self.root.update()
            self.root.after(10)

        cv.destroyAllWindows()
        cap.release()
        return



    def click_button4(self):
        cap = cv.VideoCapture(0)

        # 读取参考
        ref_person = cv.imread('ReferenceImages/1.jpg')

        person_data = init_model.object_detector(ref_person)
        person_width_in_rf = person_data[0][1]  # 得到人的像素宽度

        # 寻找焦距
        focal_person = init_model.focal_length_finder(init_model.KNOWN_DISTANCE, init_model.PERSON_WIDTH,
                                                      person_width_in_rf)

        while (True):
            ref, frame = cap.read()
            frame = cv.flip(frame, 1)  # 摄像头翻转
            img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            classes, scores, boxes = init_model.model2.detect(frame, init_model.CONFIDENCE_THRESHOLD,
                                                              init_model.NMS_THRESHOLD)

            for (classid, score, box) in zip(classes, scores, boxes):
                if classid == 0:
                    color = init_model.COLORS[int(classid) % len(init_model.COLORS)]
                    label = "%s" % (init_model.class_names[classid[0]])

                    # 在对象上绘制矩形并在对象上添加标签
                    cv.rectangle(frame, box, color, 2)
                    frame = init_model.cv2_image_add_Chinese(frame, label, box[0] + 5, box[1] - 28, color, 20)
                    distance = init_model.distance_finder(focal_person, init_model.PERSON_WIDTH, box[2])
                    x, y = box[0], box[1]
                    frame = init_model.cv2_image_add_Chinese(frame, f'距离: {round(distance, 2)} m', x + 4, y + 6, (0, 0, 255), 20)



            output = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
            pilImage = Image.fromarray(output)
            tkImage = ImageTk.PhotoImage(image=pilImage)
            self.work_output_cv.create_image(0, 0, anchor='nw', image=tkImage)
            self.root.update()
            self.root.after(10)

        cv.destroyAllWindows()
        cap.release()
        return



    def click_button5(self):
        cap = cv.VideoCapture(0)
        while (True):
            ref, frame = cap.read()
            frame = cv.flip(frame, 1)  # 摄像头翻转

            frameRGB = np.array(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

            # 代入1号模型

            input1 = init_model.trans3(frameRGB).unsqueeze(0).cuda()
            output = init_model.model3(input1)
            labels = output[0]['labels']
            boxes = output[0]['boxes']
            scores = output[0]['scores']

            # 初始化随机种子
            random.seed(2)
            color = np.random.randint(45, 255, (91, 1, 3))

            img = init_model.ShowPicResult(frame, scores, labels, boxes, color)

            cvimage = cv.cvtColor(img, cv.COLOR_BGR2RGBA)

            pilImage = Image.fromarray(cvimage)
            tkImage = ImageTk.PhotoImage(image=pilImage)
            self.work_output_cv.create_image(0, 0, anchor='nw', image=tkImage)
            self.root.update()
            self.root.after(10)
        return



    def __call__(self):
        self.init_cv()
        self.init_button()


