import cv2
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time 
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context


x,y = fetch_openml("mnist_784", version=1, return_X_y=True)
print(pd.Series(y).value_counts())
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
nclasses = len(classes)

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 9, train_size = 7500, test_size = 2500)
x_trainscale = x_train/255.0
x_testscale = x_test/255.0

clf = LogisticRegression(solver="saga", multi_class="multinomial").fit(x_trainscale,  y_train)

y_predict = clf.predict(x_testscale)
accuracy = accuracy_score(y_test, y_predict)
print(accuracy)

cap = cv2.VideoCapture(0)
while True:
    try:
        ret,frame = cap.read()
        grey=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = grey.shape
        upperleft = (int(width/2-56), int(height/2-56))
        bottomright = (int(width/2+56), int(height/2+56))
        cv2.rectangle(grey, upperleft, bottomright, (0,255,0), 2)
        roi = grey[upperleft[1]:bottomright[1], upperleft[0]:bottomright[0]]
        impil = Image.fromarray(roi)
        imagebw = impil.convert("L")
        imbwresized = imagebw.resize((28,28), Image.ANTIALIAS)
        imbwresizedinverted = PIL.ImageOps.invert(imbwresized)
        pixelfilter = 20
        minpixel = np.percentile(imbwresizedinverted, pixelfilter)
        imbwresizedinvertedscaled = np.clip(imbwresizedinverted-minpixel, 0, 255)
        maxpixel = np.max(imbwresizedinverted)
        imbwresizedinvertedscaled = np.asarray(imbwresizedinvertedscaled)/maxpixel
        testsample = np.array(imbwresizedinvertedscaled).reshape(1,784)
        testpredict = clf.predict(testsample)
        print(testpredict)
        cv2.imshow("frame", grey)
        if cv2.waitKey(1) & 0xFF == ord("q") :
            break
    except Excpetion as e :
        pass
cap.release()
cv2.destroyAllWindows()






