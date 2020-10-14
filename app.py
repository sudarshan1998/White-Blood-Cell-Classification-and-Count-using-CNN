from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from tensorflow.keras.models import Sequential


import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from google.colab.patches import cv2_imshow
import pandas



# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow as tf
from functools import reduce

# Define a flask app
app = Flask(__name__)

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']


global graph
graph = tf.get_default_graph()

# Model saved with Keras model.save()
# MODEL_PATH = '/home/sudarshan/wbcmodel.h5'

# Load your trained model
# model = load_model(MODEL_PATH, compile = False)
# model._make_predict_function()      


print('Model loaded. Check http://127.0.0.1:5000/')

def wbc_count(img_path):
	image = cv2.imread(img_path)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cv2.imwrite('/home/sudarshan/finalYear/static/gray.png',gray)

	blurM = cv2.medianBlur(gray, 5)
	cv2.imwrite('/home/sudarshan/finalYear/static/blurM.png',blurM)

	blurG = cv2.GaussianBlur(gray,(9,9), 0)
	cv2.imwrite('/home/sudarshan/finalYear/static/blurG.png',blurG)

	histoNorm = cv2.equalizeHist(gray)
	cv2.imwrite('/home/sudarshan/finalYear/static/histoNorm.png',histoNorm)

	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	claheNorm = clahe.apply(gray)
	cv2.imwrite('/home/sudarshan/finalYear/static/claheNorm.png',claheNorm)

	def pixelVal(pix, r1, s1, r2, s2): 
	    if (0 <= pix and pix <= r1): 
	        return (s1 / r1)*pix 
	    elif (r1 < pix and pix <= r2): 
	        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1 
	    else: 
	        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2 

	r1 = 70
	s1 = 0
	r2 = 200
	s2 = 255

	pixelVal_vec = np.vectorize(pixelVal) 
	  
	# Apply contrast stretching. 
	contrast_stretched = pixelVal_vec(gray, r1, s1, r2, s2) 
	contrast_stretched_blurM = pixelVal_vec(blurM, r1, s1, r2, s2) 
	  
	# Save edited images
	cv2.imwrite('/home/sudarshan/finalYear/static/contrast_stretch.png', contrast_stretched)
	cv2.imwrite('/home/sudarshan/finalYear/static/contrast_stretch_blurM.png', contrast_stretched_blurM)

	#edge detection using canny edge detector
	edge = cv2.Canny(gray,100,200)
	cv2.imwrite('/home/sudarshan/finalYear/static/edge.png',edge)
	edgeG = cv2.Canny(blurG,100,200)
	cv2.imwrite('/home/sudarshan/finalYear/static/edgeG.png',edgeG)
	edgeM = cv2.Canny(blurM,100,200)
	cv2.imwrite('/home/sudarshan/finalYear/static/edgeM.png',edgeM)



	fig = plt.figure()
	a = fig.add_subplot(3, 3, 1)
	imgplot = plt.imshow(gray, cmap='Greys_r')
	a.set_title('a. Gray Scale Image')
	a = fig.add_subplot(3, 3, 2)
	imgplot = plt.imshow(blurM, cmap='Greys_r')
	a.set_title('b. Output image')
	

	img = cv2.imread('/home/sudarshan/finalYear/static/edge.png',0)

	kernel = np.ones((5,5),np.uint8)
	dilation = cv2.dilate(img,kernel,iterations = 1)
	closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

	th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
	            cv2.THRESH_BINARY,11,2)
	th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	            cv2.THRESH_BINARY,11,2)
	#Otsu's thresholding
	ret4,th4 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# Initialize the list
	Pipe_count, x_count, y_count = [], [], []

	display = cv2.imread("/home/sudarshan/finalYear/static/claheNorm_blurM.png")

	image =th3

	circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.2,20,param1=50,param2=28,minRadius=1,maxRadius=20)

	#circle detection and labeling using hough transformation 
	if circles is not None:
	        # convert the (x, y) coordinates and radius of the circles to integers
	        circles = np.round(circles[0, :]).astype("int")

	        # loop over the (x, y) coordinates and radius of the circles
	        for (x, y, r) in circles:

	                cv2.circle(display, (x, y), r, (0, 255, 0), 2)
	                cv2.rectangle(display, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255),-1)
	                Pipe_count.append(r)
	                x_count.append(x)
	                y_count.append(y)
	 


       # show the output image

	cv2.waitKey(0)

	count = str(len(Pipe_count))
	imgplot = plt.imshow(image)
	plt.show()

	# listToStr = ' '.join([str(elem) for elem in Pipe_count]) 
  
	# print(listToStr)  
	# count = np.array(Pipe_count)
	# print(type(Pipe_count))
	# print(Pipe_count)  # Total number of radius
	# print(x_count)     # X co-ordinate of circle
	# print(y_count)     # Y co-ordinate of circle
	return count


def model_predict(img_path):
    img = image.load_img(img_path, target_size=(64, 64, 3))

    # Preprocessing the image
    img = np.array(img)
    # x = np.true_divide(x, 255)
    img = np.expand_dims(img, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x)
    # preds = model.predict(x)

    
    return img


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = model_predict(file_path)
        # wbc_counting = wbc_count(file_path)
        data = request.form.get('choose')
        if data == 'classify':
        	with graph.as_default():
        		model = load_model('/home/sudarshan/wbcmodel.h5', compile = False)
        		new_preds = model.predict(img)
        	a = np.array(new_preds)
        	i = list(a[0])
        	lists = list(a)
        	flatten_list = reduce(lambda z, y :z + y, lists)
        	listToStr = ' '.join([str(elem) for elem in flatten_list]) 
        	strings = listToStr.split()
        	totalResult = []
        	# print(listToStr) 
        	

        	index = i.index(max(i))
        	if index == 0:
        		result = 'eosinophil'
        	elif index == 1:
        		result = 'lymphocyte'
        	elif index == 2:
        		result = 'monocyte'
        	elif index == 3:
        		result = 'neutrophil'
        	else:
        		result = 'none'

        	totalResult.append(result)
        	totalResult.append(strings)
        	print(strings[0])
        	return {'result': result, 'probability': strings}
        elif data == 'count':
        	wbc_counting = wbc_count(file_path)
        	# print(wbc_counting)
        	return wbc_counting

    return None


if __name__ == '__main__':
    app.run(debug=True)

