#-*- coding: utf-8 -*-
#编写人：邓轩
#开发时间：2021/9/12 13:57

import torch
import torchvision.transforms as T
import torchvision
import tkinter as tk

import cv2 as cv

import numpy as np
from PIL import Image,ImageDraw,ImageFont

# 加载所需模型

class init_model:
    def __init__(self):

        # 深度图模型
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid").eval().cuda()  # GPU驱动
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = self.midas_transforms.small_transform

        # 物体识别YOLOV4模型

        yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

        yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

        self.model2 = cv.dnn_DetectionModel(yoloNet)
        self.model2.setInputParams(size=(480, 640), scale=1/255, swapRB=True)

        # 物体识别MASK_RCNN模型

        self.model3 = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model3 = self.model3.eval().cuda()  # GPU驱动

        self.trans3 = T.Compose([
            T.ToTensor(),
            T.Normalize([0, 0, 0], [1, 1, 1])
        ])

        # 初始化人物参数    单位：米
        self.KNOWN_DISTANCE = 1.6
        self.PERSON_WIDTH = 0.42

        # 目标探测器常数
        self.CONFIDENCE_THRESHOLD = 0.4  # 物体识别的最低得分
        self.NMS_THRESHOLD = 0.3  # 从重叠度高于阈值的两个框中删掉置信度低的那个

        # 定义颜色
        self.COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
        self.GREEN = (0, 255, 0)
        self.BLACK = (0, 0, 0)
        # 定义字体
        self.FONTS = cv.FONT_HERSHEY_COMPLEX

        self.class_names = []




    def object_detector(self,image):
        '''
        目标检测函数
        '''
        classes, scores, boxes = self.model2.detect(image, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        # 创建空列表以添加对象数据
        data_list =[]
        for (classid, score, box) in zip(classes, scores, boxes):
            # 根据对象的类id定义每个对象的颜色
            color= self.COLORS[int(classid) % len(self.COLORS)]

            label = "%s" % (self.class_names[classid[0]])

            if classid ==0: # 如果检测到人
                data_list.append([self.class_names[classid[0]], box[2], (box[0], box[1]-2)])
        return data_list

    def focal_length_finder(self,measured_distance, real_width, width_in_rf):
        '''
        计算焦距： 焦距=（像素宽度 * 实际距离）/ 实际宽度
        '''
        focal_length = (width_in_rf * measured_distance) / real_width
        return focal_length

    def distance_finder(self,focal_length, real_object_width, width_in_frmae):
        '''
        计算距离： 实际距离 = （实际宽度 * 焦距） / 像素宽度
        '''
        distance = (real_object_width * focal_length) / width_in_frmae
        return distance

    def get_lable(self):
        with open("classes.txt", "r", encoding='utf-8') as f:
            self.class_names = [cname.strip() for cname in f.readlines()]

    #########################################################################
    # 道路检测
    def canyEdgeDetector(self,image):
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        edged = cv.Canny(image, 150, 160)
        return edged

    def getROI(self,image):
        height = image.shape[0]
        width = image.shape[1]
        triangle = np.array([[(0, height + 200), (width, height + 200), (int(width / 2) - 10, int(height / 2) + 30)]])
        black_image = np.zeros_like(image)
        mask = cv.fillPoly(black_image, triangle, 255)
        masked_image = cv.bitwise_and(image, mask)
        return masked_image

    def getLines(self,image):
        lines = cv.HoughLinesP(image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        return lines

    def displayLines(self,image, lines):
        if lines is not None:
            height = image.shape[0]
            width = image.shape[1]

            x1, y1, x2, y2 = lines[0].reshape(4)

            x3, y3, x4, y4 = lines[1].reshape(4)

            x0 = int(((x3 - x4) * (x2 * y1 - x1 * y2) - (x1 - x2) * (x4 * y3 - x3 * y4)) / (
                    (x3 - x4) * (y1 - y2) - (x1 - x2) * (y3 - y4)))
            y0 = int(((y3 - y4) * (y2 * x1 - y1 * x2) - (y1 - y2) * (y4 * x3 - y3 * x4)) / (
                    (y3 - y4) * (x1 - x2) - (y1 - y2) * (x3 - x4)))


            black_image = np.zeros_like(image)

            black_image = cv.fillConvexPoly(black_image, np.array([[x1, y1], [x3, y3], [x0, y0]]), (255, 20, 125))

            result = cv.addWeighted(image, 1, black_image, 0.5, 0)

        return result

    def getLineCoordinatesFromParameters(self,image, line_parameters):
        slope = line_parameters[0]
        intercept = line_parameters[1]
        y1 = image.shape[0]
        y2 = int(y1 * (3.4 / 5))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    def getSmoothLines(self,image, lines):
        left_fit = []
        right_fit = []

        right_line = [0, 0, 0, 0]
        left_line = [0, 0, 0, 0]

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]        \


            if slope <= 0:
                left_fit.append((slope, intercept))
                left_fit_average = np.average(left_fit, axis=0)
                left_line = self.getLineCoordinatesFromParameters(image, left_fit_average)
            else:
                right_fit.append((slope, intercept))
                right_fit_average = np.average(right_fit, axis=0)
                right_line = self.getLineCoordinatesFromParameters(image, right_fit_average)
        return np.array([left_line, right_line])

    ###################################################################################################################

    ###################################################################################################################
    # 测距
    def cv2_image_add_Chinese(self,img, text, left, top, color, size):
        '''
        在opencv中添加中文的函数
        '''
        img = Image.fromarray(img)

        draw = ImageDraw.Draw(img)
        font_style = ImageFont.truetype("font/HGCH_CNKI.TTF", size, encoding="utf-8", index=0)
        draw.text((left, top), text, color, font=font_style)

        return np.asarray(img)



    def CentralShow(self,img):
        """
        最终结果显示函数
        """
        return img

    def GetClass(self):  # 读取标签字典
        id2name = {}
        f = open('coco_name', 'r+', encoding='utf-8')
        for idx, line in enumerate(f):
            id2name[idx + 1] = line.strip()
        return id2name

    def get_distance(self,cv_mat, x0, y0):

        K = 10 # 纠正系数
        H = 1  # 摄像头高度为1m
        DMIN = 0  # 摄像头最近视角距离
        DMAX = 1000  # 摄像头最远视角距离
        beta = math.pi / 90 * 150  # 水平视场角
        height = cv_mat.shape[0]  # 图像高度
        width = cv_mat.shape[1]  # 图像宽度

        alpha = math.atan(DMIN / H*K)  # 俯仰角
        theta = math.atan(DMAX / H*K) - alpha  # 垂直视场角

        # y0 = 10

        delta_theta = (height - y0) / height * theta  # 步进小角度
        y1 = H * K * math.tan(alpha + delta_theta)  # 距离摄像头的垂直距离
        x1 = 2 * DMAX * math.tan(beta / 2) * (x0 - width / 2) / width  # 距离摄像头的水平距离

        distance = math.sqrt(y1 ** 2 + x1 ** 2)  # 距离摄像头的总距离

        return y1

    def ShowPicResult(self,cv_mat, scores, labels, boxes, colors):
        id2name = self.GetClass()

        cv_mat_half = cv_mat[cv_mat.shape[0] * 9 // 20: cv_mat.shape[0], :]

        edged_image = self.canyEdgeDetector(cv_mat)  # Step 1
        roi_image = self.getROI(edged_image)  # Step 2
        lines = self.getLines(roi_image)  # Step 3
        smooth_lines = self.getSmoothLines(cv_mat, lines)  # Step 5
        cv_mat = self.displayLines(cv_mat, smooth_lines)  # Step 4

        for score, label, box in zip(scores, labels, boxes):

            if score > 0.9  and label.item() == 3:
                x1, y1, x2, y2 = box

                color = colors[label.item()].squeeze(0)
                color = tuple(int(a) for a in color)
                bbox_x = (x1 + x2) / 2  # 包容盒下边界中点x坐标
                bbox_y = y2  # 包容盒下边界中点y坐标


                h = self.get_distance(cv_mat_half, cv_mat.shape[1] - bbox_x, bbox_y - cv_mat_half.shape[0])

                string = id2name[label.item()] + " " + str('%.2f' % h) + 'm'
                string = str('%.2f' % h) + 'm'

                cv_mat = cv.line(cv_mat, (int(cv_mat.shape[1] / 2), int(cv_mat.shape[0])),
                                  (int((x1 + x2) / 2), int(y2)), (0, 255, 0), 2)

                cv_mat = self.cv2_image_add_Chinese(cv_mat, string, int((x1 + x2) / 2) + int((x1 - x2) / 5), int(y1),
                                               (0, 0, 255), int(int(x2 - x1) / 4) + 15)
                cv_mat = cv.rectangle(cv_mat, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        img = self.CentralShow(cv_mat)
        return img


    def __call__(self):
        self.get_lable()

