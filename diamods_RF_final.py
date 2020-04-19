#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib
import os,sys,string,re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# split
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
from sklearn.ensemble import RandomForestClassifier
#創建GUI窗口打開圖像 顯示在窗口中
from PIL import Image, ImageTk # 導入圖像處理函數庫
import matplotlib.pyplot as plt # 繪圖庫
import numpy as np
import tkinter as tk           # 介面
import tkinter.filedialog
import os.path
global img_png

# 設定窗口大小命名
window = tk.Tk()
window.title('DM 五 RandomForestClassifier')
window.geometry('850x500')
var = tk.StringVar()    

#輸入層數
carat_input = tk.Label(window, text = "Input carat(0.2-5.01)：")
carat_input.place(x = 100, y = 42)
carat_text = 'a'
carat_text = tk.Entry(window, show = None, width = 5) #輸入框
carat_text.place(x = 250, y = 40)

cut_text = tk.Label(window,text = "Input cut: ")
cut_text.place(x = 100, y = 100)
cut_input = tk.StringVar(window)
cut_input.set("Fair(5)") # default value
cut = tk.OptionMenu(window, cut_input, "Fair(5)", "Good(4)", "Very Good(3)", "Premium(2)","Ideal(1)")
cut.place(x = 230, y = 100)

color_text = tk.Label(window,text = "Input color: ")
color_text.place(x = 100, y = 150)
color_input = tk.StringVar(window)
color_input.set("J(7)") # default value
color = tk.OptionMenu(window, color_input, "J(7)", "I(6)", "H(5)", "G(4)","F(3)","E(2)","D(1)")
color.place(x = 230, y = 150)

clarity_text = tk.Label(window,text = "Input clarity: ")
clarity_text.place(x = 100, y = 200)
clarity_input = tk.StringVar(window)
clarity_input.set("II(8)") # default value
clarity = tk.OptionMenu(window, clarity_input, "II(8)", "SIII(7)", "SII(6)", "VSII(5)","VSI(4)","VVSII(3)","VVSI(2)","IF(1)")
clarity.place(x = 230, y = 200)

var_price = tk.StringVar()
var_range = tk.StringVar()

img = Image.open('/Users/kos408/Data mining/new.png')
img_range = ImageTk.PhotoImage(img)
label_img = tk.Label(window, image = img_range)
label_img.place(x = 380, y = 50)  # 第一張圖的位置

# read the data
data = pd.read_csv('diamonds_data.csv')
data = data.drop(data.columns[0],axis=1)

def KNN_model():
    carat = carat_text.get() #抓取輸入的層數
    carat = float(carat)
    cut = str(cut_input.get())
    cut = re.sub("\D", "", cut) #只保留數字
    color = str(color_input.get())
    color = re.sub("\D", "", color) 
    clarity = str(clarity_input.get())
    clarity = re.sub("\D", "", clarity) 
    price = int(carat*100*31)
    print(carat)
    print(cut)
    print(color)
    print(clarity)

     #連續carat+二值化3c
    sampled_data=pd.concat([data[data['priceperpoint']==4].sample(7000),data[data['priceperpoint']==3].sample(7000),data[data['priceperpoint']==2].sample(7000),data[data['priceperpoint']==1].sample(7000)],ignore_index=True)
    
    X = sampled_data[['carat','cut','color','clarity']]
    y = sampled_data['priceperpoint']
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.15, random_state=0, stratify=None)

    RF_Ori = RandomForestClassifier(criterion='gini', random_state=5, n_estimators=200, min_samples_split=50, oob_score=True)
    RF_Ori.fit(X_train, y_train)
    
    # define one new instance
    X_new = [[carat,cut,color,clarity]]
    # make a prediction
    y_new = str(RF_Ori.predict(X_new))
    y_show = re.sub("\D", "", y_new) 
    price2 = 0
    price_range = ''
    if y_show == "1":
        y_show = '預測為 低價位 鑽石'
        price1 = price*21
        price_range = str('最高'+str(price1))
    elif y_show == "2":
        y_show = '預測為 中低價位 鑽石'
        price1 = price*21
        price2 = price*42
        price_range = str(str(price1)+'至'+str(price2))
    elif y_show == "3":
        y_show = '預測為 中高價位 鑽石'
        price1 = price*42
        price2 = price*63
        price_range = str(str(price1)+'至'+str(price2))
    elif y_show == "4":
        y_show = '預測為 高價位 鑽石'
        price1 = price*63
        price_range = str('最少'+str(price1))
    
    print("Predicted=%s" % (y_new))
    var_price.set('鑽石價位：'+str(price_range)+' 臺幣')
    Label_Show = tk.Label(window, textvariable = var_price, width = 30, height = 5)
    Label_Show.place(x = 80, y = 300)

    var_range.set(y_show)
    Label_Show2 = tk.Label(window, textvariable = var_range, width = 20, height = 2)
    Label_Show2.place(x = 360, y = 230)

# 創建打開影像按鈕 
btn_Open = tk.Button(window,
    text = 'Evaluate',      
    width = 13, height = 2,
    command = KNN_model)     # 執行open img
btn_Open.place(x = 100, y = 250)    # 按鈕位置
'''
Label_Show = tk.Label(window,
    textvariable = var,   # 使用 textvariable 替換 text, 文字可以做變化
    width = 70, height = 2)
Label_Show.place(x = 100, y = 70)
'''
# 運行整體窗口
window.mainloop()