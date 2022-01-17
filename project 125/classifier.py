import numpy as np 
import pandas as pd
import csv
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

#fetching the data 
x= np.load('image.npz')['arr_0']
y= pd.read_csv("labels.csv")("labels")

print(pd.Series(y).value_counts)

classes = ['A','B','C','D','E','F','G','H','I','J','K','L','N','O',
'P','R','S','T','U','V','W','X','Y','Z']

nclasses = len(classes)

#Splitting the data
X_train,X_test,y_train,y_test= train_test_split(X,y, random_state=9, train_size=3500, test_size=500)
#Scaling the date
X_train_scaled=X_train/255.0
X_test_scaled=X_test/255.0

#fitting the data into the logistic regression
clf=LogisticRegression(solver='sage',multi_class='multinomial').fit(X_train_scaled, y_train)

def get_prediction(image):
    im_pil=Image.open(image)
    image_bw=im_pil.convert('L')
    image_bw_resized=image_bw_resize(
        (22,30),
        Image.ANTIALIAS
    )
    pixel_filter=20
    min_pixel=np.percentile(image_bw_resized,pixel_filter)
    image_bw_resized_inverted_scaled=np.clip(image_bw_resized-min_pixel,0,255)
    max_pixel=np.max(image_bw_resized)
    image_bw_resized_inverted_scaled=np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample=np.array(image_bw_resized_inverted_scaled).reshape(1,660)
    test_pred=clf.predict(test_sample)
    return test_pred[0]

