#-*- coding: utf-8 -*-
#编写人：邓轩
#开发时间：2021/9/12 12:50


import tkinter as tk
from init_window import Tkwindow
from init_model import init_model

if __name__ == "__main__":

    window = tk.Tk()
    window.title('单目行车测距与多功能摄像头 v0.9                   创客杯比赛')
    window.geometry('1000x480')
    window.resizable(0, 0)                  # 防止用户调整尺寸

    app_window = Tkwindow(window)
    app_window()

    window.mainloop()