from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, Flatten, LeakyReLU
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow import keras
from tqdm import tqdm
import numpy as np
import pickle
import cv2
from PIL import Image, ImageOps
import os
from matplotlib import pyplot

class predictImage:
	def __init__(self):
		self.a = 5

	def DatasetImage(self):
		print("Dataset")
		path = "//home//adarsh//Desktop//check//data"
		image = []
		classNumber = []
		myList = os.listdir(path)
		print(myList)
		numOfClasses = len(myList)
		print(numOfClasses)
		for x in range(0, numOfClasses):
			myPicList = os.listdir(path+'/'+myList[x])
			for y in myPicList:
				u = path + '/' + myList[x] +'/'+ y
				curImg = cv2.imread(u) #having larger in size would increase computational power
				curImg = cv2.resize(curImg,(28, 28))
				image.append(curImg)
				classNumber.append(myList[x])
			#print(x,end=" ")
		image = np.array(image)
		classNumber = np.array(classNumber)
		return image, classNumber, numOfClasses
		
	def splitingDataset(self, image, classNumber):
		print("split data")
		#spliting data in between test and train
		Xtrain, Xtest, Ytrain, Ytest = train_test_split(image, classNumber, test_size=0.2)
		return Xtrain, Xtest, Ytrain, Ytest
		
	def preprocessing(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert iamge yo GRAYScale
		(thresh, img) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
		#img = cv2.equalizeHist(img)
		#img = img/255 # dividing every pixel by 255 normalization of each pixel to [0, 1]
		return img
		
	def DatasetToList(self, Xtrain, Xtest):
		# here i did 4 thing
		# 1st to convert/ preprocessing image
		# 2nd mapping preprocessing image
		# 3rd converting to list
		# 4th converting to numpy array
		print("dataset to list")
		Xtrain = np.array(list(map(self.preprocessing, Xtrain)))
		Xtest  = np.array(list(map(self.preprocessing, Xtest)))
		
		Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2], 1)
		Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], Xtest.shape[2], 1)
		#print(Xtrain.shape) # after reshape (44, 32, 32, 1)
		return Xtrain, Xtest
	
	def changesIn_Y_Set(self, Ytrain, Ytest, numOfClasses):
		print("changeIn Y set")
		datagen = ImageDataGenerator(width_shift_range=0.1, 
					height_shift_range=0.1,
					zoom_range=0.2,
					shear_range=0.1,
					rotation_range=0)
	
		datagen.fit(Xtrain)
	
		integer_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '(': 10, ')': 11, '+': 12, '-': 13, '*': 14, 'd': 15}
		Ytrain = [integer_mapping[word] for word in Ytrain]
		Ytest = [integer_mapping[word] for word in Ytest]
		
		#converting string to integer in Ytrain
		Ytrain = np.array(Ytrain)
		label_encoder = LabelEncoder()
		Ytrain = label_encoder.fit_transform(Ytrain)
		
		#converting string to integer in Ytest
		Ytest = np.array(Ytest)
		label_encoder = LabelEncoder()
		Ytest = label_encoder.fit_transform(Ytest)
		
		#to_categorical method use integer
		Ytrain = to_categorical(Ytrain, numOfClasses)
		Ytest = to_categorical(Ytest, numOfClasses)	
		return Ytrain, Ytest, datagen, integer_mapping
	
	def myModel(self, datagen, Xtrain, Ytrain, Xtest, Ytest):
		print("Model")
		model = keras.Sequential()
		
		model.add(Conv2D(256, (5, 5), input_shape=(28, 28, 1), activation='relu'))
		#model.add(LeakyReLU(alpha=0.01))
		
		model.add(Conv2D(128, (4, 4), activation='relu')) #tanh relu
		
		model.add(Conv2D(100, (3, 3), activation='relu'))
		
		model.add(MaxPooling2D(3, 3))
		
		model.add(Dropout(0.5))
		model.add(Flatten())
		
		model.add(Dense(100, activation='relu'))
		
		model.add(Dropout(0.5))
		
		#mean_squared_error
		#categorical_crossentropy
		#mean_squared_logarithmic_error
		#mean_absolute_error
		#binary_crossentropy
		#hinge
		#squared_hinge
		#sparse_categorical_crossentropy
		#kullback_leibler_divergence
		
		#model.add(LeakyReLU(alpha=0.01))
		
		model.add(Dense(numOfClasses, activation='softmax'))
		#softmax is generalization of sigmoid function
		
		model.compile(keras.optimizers.Adam(learning_rate=0.001) , loss='categorical_crossentropy', metrics=['accuracy'])
		history = model.fit_generator(
					datagen.flow(Xtrain, Ytrain, batch_size=50),
					 epochs=5,
					validation_data=(Xtest, Ytest),
					shuffle=0)
		return history, model
	
	
	def plotGraph(self, history):
		print("plot")
		#Loss
		plt.figure(1)
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.legend(['training', 'validation'])
		plt.title('Loss')
		plt.xlabel('epoch')
		
		#Accuracy
		plt.figure(2)
		plt.plot(history.history['accuracy'])
		plt.plot(history.history['val_accuracy'])
		plt.legend(['training', 'validation'])
		plt.title('Accuracy')
		plt.xlabel('epoch')
		plt.show()
	
	def evaluate(self, model, Xtest, Ytest):
		score = model.evaluate(Xtest, Ytest, verbose=0)
		print('Test Score = ', score[0])
		print('accuracy Score = ', score[1])


pl = predictImage()
image, classNumber, numOfClasses = pl.DatasetImage()
Xtrain, Xtest, Ytrain, Ytest = pl.splitingDataset(image, classNumber)
Xtrain, Xtest = pl.DatasetToList(Xtrain, Xtest)
Ytrain, Ytest, datagen, integer_mapping = pl.changesIn_Y_Set(Ytrain, Ytest, numOfClasses)
history, model = pl.myModel(datagen, Xtrain, Ytrain, Xtest, Ytest)
pl.plotGraph(history)
pl.evaluate(model, Xtest, Ytest)

"""
print("Image 41")
r = '//home//adarsh//Desktop//check//41.png'
img1 = cv2.imread(r)
img1 = cv2.resize(img1, (28, 28))
img1 = pl.preprocessing(img1)
img1 = np.reshape(img1, [1, 28, 28, 1])

classIndex = int(model.predict_classes(img1))
print("class: ",classIndex)
prediction = model.predict(img1)
#print("ppred : ", prediction)
probval = np.amax(prediction)
print(classIndex, probval)


print("Image 41")
r = '//home//adarsh//Desktop//check//41.png'
img1 = cv2.imread(r)
img1 = cv2.resize(img1, (28, 28))
img1 = pl.preprocessing(img1)
img1 = np.reshape(img1, [1, 28, 28, 1])

classIndex = int(model.predict_classes(img1))
print("class: ",classIndex)
prediction = model.predict(img1)
#print("ppred : ", prediction)
probval = np.amax(prediction)
print(classIndex, probval)

print("Image 42")
r = '//home//adarsh//Desktop//check//42.png'
img1 = cv2.imread(r)
img1 = cv2.resize(img1, (28, 28))
img1 = pl.preprocessing(img1)
img1 = np.reshape(img1, [1, 28, 28, 1])

classIndex = int(model.predict_classes(img1))
print("class: ",classIndex)
prediction = model.predict(img1)
#print("ppred : ", prediction)
probval = np.amax(prediction)
print(classIndex, probval)

print("Image 43")
r = '//home//adarsh//Desktop//check//43.png'
img1 = cv2.imread(r)
img1 = cv2.resize(img1, (28, 28))
img1 = pl.preprocessing(img1)
img1 = np.reshape(img1, [1, 28, 28, 1])

classIndex = int(model.predict_classes(img1))
print("class: ",classIndex)
prediction = model.predict(img1)
#print("ppred : ", prediction)
probval = np.amax(prediction)
print(classIndex, probval)

print("Image 44")
r = '//home//adarsh//Desktop//check//44.png'
img1 = cv2.imread(r)
img1 = cv2.resize(img1, (28, 28))
img1 = pl.preprocessing(img1)
img1 = np.reshape(img1, [1, 28, 28, 1])

classIndex = int(model.predict_classes(img1))
print("class: ",classIndex)
prediction = model.predict(img1)
#print("ppred : ", prediction)
probval = np.amax(prediction)
print(classIndex, probval)
['8', '1', '2', '+', '0', '9', '3', '7', '(', '*', '4', '6', 'd', '5', '-', ')']
"""

