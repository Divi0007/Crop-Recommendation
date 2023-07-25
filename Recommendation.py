import numpy as np
import pandas as pd
import tensorflow as tn

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
dataset=pd.read_csv('/Users/divyanshyadav/Downloads/Crop_recommendation.csv')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
le=LabelEncoder()
y[:]=le.fit_transform(y[:])
le_name=dict(zip(le.classes_,le.transform(le.classes_)))
print(le_name)

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])], remainder='passthrough')
# y=np.array(ct.fit_transform(y))
# print("hello",y)
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
X_scale = min_max_scaler.fit_transform(x)
y = np.asarray(y).astype('float32')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=.2, random_state=33)

print(x_train.shape)
print(y_train.shape)
#,input_shape=x_train[0].shape
# ann=tn.keras.models.Sequential()
# ann.add(tn.keras.layers.Dense(units=5,activation='relu'))
# ann.add(tn.keras.layers.Dense(units=6,activation='relu'))
# ann.add(tn.keras.layers.Dense(units=1,activation='softmax'))
# ann.compile(optimizer='adam',loss='mse', metrics=['accuracy'])
# ann.fit(x_train,y_train,batch_size=20,epochs =25)
# print(ann.predict(sc.transform([[-0.64463185,2.00512125,2.96466347, -0.81644358 , 0.86301256, -0.4785779,0.2369273]])))
ann=tn.keras.models.Sequential()
ann.add(tn.keras.layers.Dense(units=44,activation='elu'))
ann.add(tn.keras.layers.Dense(units=44,activation='elu'))
ann.add(tn.keras.layers.Dense(units=22,activation='softmax'))
ann.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist=ann.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=30,epochs =100)
y_pre=ann.predict(x_test)
print(y_pre.size)
print(y_pre)
a=[]
for i in y_pre:
    max = 0
    c=None
    for j in range(len(i)):
        if i[j]>max:
            max=i[j]
            c=j
    a.append(c)
#
# print(a)
print(accuracy_score(y_test,a)*100)
print(y_test[35])
print(y_pre[35])

import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
w1=0
w2=""


def fr(frame):
    frame.tkraise()
def but():
    n=n1.get()
    p=p2.get()
    k=k1.get()
    t=temp.get()
    h=hum.get()
    ph1=ph.get()
    r=ra.get()
    test = [n, p, k, t, h, ph1, r]
    y_pre = ann.predict([test])
    a = []
    for i in y_pre:
        max = 0
        c = None
        for j in range(len(i)):
            if i[j] > max:
                max = i[j]
                c = j
        a.append(c)
    print(a)
    list1 = []
    global w1,w2
    for i in a:
        for j in le_name:
            if le_name[j] == i:
                list1.append(j)
    w1=a[0]
    w2=list1[0]
    aq1 = tk.Label(p1, text="             ", bg='light blue', font=("Times New Roman", 20))
    aq2 = tk.Label(p1, text="                  ", bg='light blue', font=("Times New Roman", 25))
    aq2.place(relx=.25, rely=.75)
    aq1.place(relx=.25, rely=.65)
    aq1 = tk.Label(p1, text=w1, bg='light blue', font=("Times New Roman", 20))
    aq2 = tk.Label(p1, text=w2, bg='light blue', font=("Times New Roman", 20))
    # aq1.place(relx=.25, rely=.65)
    aq2.place(relx=.25, rely=.75)

    if w1==0:
        path="/Users/divyanshyadav/Downloads/crop_pic/apple.jpg"
        q = Image.open(path)
        img1 = q.resize((200, 200))
        img = ImageTk.PhotoImage(img1)
        panel = Label(p1, image=img)
        panel.photo = img
        panel.place(relx=.5, rely=.65)
    elif w1==1:
        path="/Users/divyanshyadav/Downloads/crop_pic/banana.jpg"
        q = Image.open(path)
        img1 = q.resize((200, 200))
        img = ImageTk.PhotoImage(img1)
        panel = Label(p1, image=img)
        panel.photo = img
        panel.place(relx=.5, rely=.65)
    elif w1 == 2:
        path="/Users/divyanshyadav/Downloads/crop_pic/black-gram.jpg"
        q = Image.open(path)
        img1 = q.resize((200, 200))
        img = ImageTk.PhotoImage(img1)
        panel = Label(p1, image=img)
        panel.photo = img
        panel.place(relx=.5, rely=.65)
    elif w1 == 3:
        path="/Users/divyanshyadav/Downloads/crop_pic/chickpeas.jpg"
        q = Image.open(path)
        img1 = q.resize((200, 200))
        img = ImageTk.PhotoImage(img1)
        panel = Label(p1, image=img)
        panel.photo = img
        panel.place(relx=.5, rely=.65)
    elif w1 == 4:
        path="/Users/divyanshyadav/Downloads/crop_pic/COCONUT.jpg"
        q = Image.open(path)
        img1 = q.resize((200, 200))
        img = ImageTk.PhotoImage(img1)
        panel = Label(p1, image=img)
        panel.photo = img
        panel.place(relx=.5, rely=.65)
    elif w1 == 5:
        path="/Users/divyanshyadav/Downloads/crop_pic/coffee.jpg"
        q = Image.open(path)
        img1 = q.resize((200, 200))
        img = ImageTk.PhotoImage(img1)
        panel = Label(p1, image=img)
        panel.photo = img
        panel.place(relx=.5, rely=.65)
    elif w1 == 6:
        path="/Users/divyanshyadav/Downloads/crop_pic/cotton.jpeg"
        q = Image.open(path)
        img1 = q.resize((200, 200))
        img = ImageTk.PhotoImage(img1)
        panel = Label(p1, image=img)
        panel.photo = img
        panel.place(relx=.5, rely=.65)
    elif w1 == 7:
        path="/Users/divyanshyadav/Downloads/crop_pic/grapes.jpg"
        q = Image.open(path)
        img1 = q.resize((200, 200))
        img = ImageTk.PhotoImage(img1)
        panel = Label(p1, image=img)
        panel.photo = img
        panel.place(relx=.5, rely=.65)
    elif w1 == 8:
        path="/Users/divyanshyadav/Downloads/crop_pic/jute.jpg"
        q = Image.open(path)
        img1 = q.resize((200, 200))
        img = ImageTk.PhotoImage(img1)
        panel = Label(p1, image=img)
        panel.photo = img
        panel.place(relx=.5, rely=.65)
    elif w1 == 9:
        path="/Users/divyanshyadav/Downloads/crop_pic/kidney_beans.jpg"
        q = Image.open(path)
        img1 = q.resize((200, 200))
        img = ImageTk.PhotoImage(img1)
        panel = Label(p1, image=img)
        panel.photo = img
        panel.place(relx=.5, rely=.65)
    elif w1 == 10:
        path="/Users/divyanshyadav/Downloads/crop_pic/lentils.jpg"
        q = Image.open(path)
        img1 = q.resize((200, 200))
        img = ImageTk.PhotoImage(img1)
        panel = Label(p1, image=img)
        panel.photo = img
        panel.place(relx=.5, rely=.65)
    elif w1 == 11:
        path="/Users/divyanshyadav/Downloads/crop_pic/maize.jpg"
        q = Image.open(path)
        img1 = q.resize((200, 200))
        img = ImageTk.PhotoImage(img1)
        panel = Label(p1, image=img)
        panel.photo = img
        panel.place(relx=.5, rely=.65)
    elif w1 == 12:
        path="/Users/divyanshyadav/Downloads/crop_pic/Mango.jpg"
        q = Image.open(path)
        img1 = q.resize((200, 200))
        img = ImageTk.PhotoImage(img1)
        panel = Label(p1, image=img)
        panel.photo = img
        panel.place(relx=.5, rely=.65)
    elif w1 == 13:
        path="/Users/divyanshyadav/Downloads/crop_pic/mothbeans.jpg"
        q = Image.open(path)
        img1 = q.resize((200, 200))
        img = ImageTk.PhotoImage(img1)
        panel = Label(p1, image=img)
        panel.photo = img
        panel.place(relx=.5, rely=.65)
    elif w1 == 14:
        path="/Users/divyanshyadav/Downloads/crop_pic/mung_beans.jpg"
        q = Image.open(path)
        img1 = q.resize((200, 200))
        img = ImageTk.PhotoImage(img1)
        panel = Label(p1, image=img)
        panel.photo = img
        panel.place(relx=.5, rely=.65)
    elif w1 == 15:
        path="/Users/divyanshyadav/Downloads/crop_pic/muskmelon.jpg"
        q = Image.open(path)
        img1 = q.resize((200, 200))
        img = ImageTk.PhotoImage(img1)
        panel = Label(p1, image=img)
        panel.photo = img
        panel.place(relx=.5, rely=.65)
    elif w1 == 16:
        path="/Users/divyanshyadav/Downloads/crop_pic/orange.jpg"
        q = Image.open(path)
        img1 = q.resize((200, 200))
        img = ImageTk.PhotoImage(img1)
        panel = Label(p1, image=img)
        panel.photo = img
        panel.place(relx=.5, rely=.65)
    elif w1 == 17:
        path="/Users/divyanshyadav/Downloads/crop_pic/pigeon_peas.jpg"
        q = Image.open(path)
        img1 = q.resize((200, 200))
        img = ImageTk.PhotoImage(img1)
        panel = Label(p1, image=img)
        panel.photo = img
        panel.place(relx=.5, rely=.65)
    elif w1 == 18:
        path="/Users/divyanshyadav/Downloads/crop_pic/pomegranate.jpg"
        q = Image.open(path)
        img1 = q.resize((200, 200))
        img = ImageTk.PhotoImage(img1)
        panel = Label(p1, image=img)
        panel.photo = img
        panel.place(relx=.5, rely=.65)
    elif w1 == 19:
        path="/Users/divyanshyadav/Downloads/crop_pic/rice.jpg"
        q = Image.open(path)
        img1 = q.resize((200, 200))
        img = ImageTk.PhotoImage(img1)
        panel = Label(p1, image=img)
        panel.photo = img
        panel.place(relx=.5, rely=.65)
    elif w1 == 20:
        path="/Users/divyanshyadav/Downloads/crop_pic/watermelon.jpg"
        q = Image.open(path)
        img1 = q.resize((200, 200))
        img = ImageTk.PhotoImage(img1)
        panel = Label(p1, image=img)
        panel.photo = img
        panel.place(relx=.5, rely=.65)



win=tk.Tk()
win.title("Crop Recommender")
win.geometry('600x550')
win.rowconfigure(0, weight=1)
win.columnconfigure(0, weight=1)
n1= tk.DoubleVar()
p2 = tk.DoubleVar()
k1 = tk.DoubleVar()
temp = tk.DoubleVar()
hum= tk.DoubleVar()
ph = tk.DoubleVar()
ra = tk.DoubleVar()
p1 = tk.Frame(win,width=200, height=100,bg='light blue')
p1.grid(row=0, column=0, sticky='nsew')

h1=tk.Label(p1,text="CROP RECOMMENDATION",bg='light blue', font=("Times New Roman", 20)).place(relx=.25, rely=.05)

q1=tk.Label(p1,text="NITROGEN : ",bg='light blue', font=("Times New Roman", 20)).place(relx=.15, rely=.15)
e1=tk.Entry(p1, textvariable=n1, width=25,highlightbackground='light blue').place(relx=.399, rely=.15)

q2=tk.Label(p1,text="PHOSPHORUS : ",bg='light blue', font=("Times New Roman", 20)).place(relx=.15, rely=.2)
e2=tk.Entry(p1, textvariable=p2, width=25,highlightbackground='light blue').place(relx=.399, rely=.2)

q3=tk.Label(p1,text="POTASSIUM : ",bg='light blue', font=("Times New Roman", 20)).place(relx=.15, rely=.25)
e3=tk.Entry(p1, textvariable=k1, width=25,highlightbackground='light blue').place(relx=.399, rely=.25)

q4=tk.Label(p1,text="TEMPERATURE : ",bg='light blue', font=("Times New Roman", 20)).place(relx=.15, rely=.3)
e4=tk.Entry(p1, textvariable=temp, width=25,highlightbackground='light blue').place(relx=.399, rely=.3)

q5=tk.Label(p1,text="HUMIDITY : ",bg='light blue', font=("Times New Roman", 20)).place(relx=.15, rely=.35)
e5=tk.Entry(p1, textvariable=hum, width=25,highlightbackground='light blue').place(relx=.399, rely=.35)

q6=tk.Label(p1,text="PH : ",bg='light blue', font=("Times New Roman", 20)).place(relx=.15, rely=.4)
e6=tk.Entry(p1, textvariable=ph, width=25,highlightbackground='light blue').place(relx=.399, rely=.4)

q7=tk.Label(p1,text="RAINFALL : ",bg='light blue', font=("Times New Roman", 20)).place(relx=.15, rely=.45)
e7=tk.Entry(p1, textvariable=ra, width=25,highlightbackground='light blue').place(relx=.399, rely=.45)

b1=tk.Button(p1,text="SUBMIT",width=25,height=2,highlightbackground='light blue',fg="yellow",command=lambda:but()).place(relx=.25, rely=.55)
# lbl_result= tk.Label(p1, text="jab").grid(row=1,column=5)

fr(p1)
win.mainloop()
