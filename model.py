from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers import Dense, Activation, Flatten
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
import os

def DatasetImage():
	path = "Dataset Directory"
	image = []
	classNumber = []
		
	myList = os.listdir(path)
	
	numOfClasses = len(myList)
	for x in range(0, numOfClasses):
		myPicList = os.listdir(path+'/'+myList[x])
		for y in myPicList:
			u = path + '/' + myList[x] +'/'+ y
			curImg = cv2.imread(u) # having larger in size would increase computational power
			curImg = cv2.resize(curImg,(28, 28))
			image.append(curImg)
			classNumber.append(myList[x])

	image = np.array(image)
	classNumber = np.array(classNumber)
	return image, classNumber, numOfClasses

def splitingDataset(image):

	Xtrain, Xtest, Ytrain, Ytest = train_test_split(image, classNumber, test_size=0.2)
	Xtrain, Xvalidation, Ytrain, Yvalidation = train_test_split(Xtrain, Ytrain, test_size=0.2)
	return Xtrain, Xtest, Ytrain, Ytest, Xvalidation, Yvalidation

def preprocessing(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert color iamge to GRAYScale
	img = cv2.equalizeHist(img)
	img = img/255 # dividing every pixel by 255
	return img

def DatasetToList(Xtrain, Xtest, Xvalidation):
	Xtrain = np.array(list(map(preprocessing, Xtrain)))
	Xtest  = np.array(list(map(preprocessing, Xtest)))
	Xvalidation = np.array(list(map(preprocessing, Xvalidation)))
	
	Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2], 1)
	Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], Xtest.shape[2], 1)
	Xvalidation = Xvalidation.reshape(Xvalidation.shape[0], Xvalidation.shape[1], 				Xvalidation.shape[2], 1)
	return Xtrain, Xtest, Xvalidation

def changesIn_Y_Set(Xtrain, Ytrain, Ytest, Yvalidation, numOfClasses):
	datagen = ImageDataGenerator(  width_shift_range=0.1, 
				height_shift_range=0.1,
				zoom_range=0.2,
				shear_range=0.1,
				rotation_range=10)

	datagen.fit(Xtrain)

	integer_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,'6': 6, '7': 7, '8': 8, '9': 9}
	Ytrain = [integer_mapping[word] for word in Ytrain]
	Ytest = [integer_mapping[word] for word in Ytest]
	Yvalidation = [integer_mapping[word] for word in Yvalidation]
	
	Ytrain = np.array(Ytrain)
	label_encoder = LabelEncoder()
	Ytrain = label_encoder.fit_transform(Ytrain)
	
	Ytest = np.array(Ytest)
	label_encoder = LabelEncoder()
	Ytest = label_encoder.fit_transform(Ytest)
	
	Yvalidation = np.array(Yvalidation)
	label_encoder = LabelEncoder()
	Yvalidation = label_encoder.fit_transform(Yvalidation)
	
	Ytrain = to_categorical(Ytrain, numOfClasses)
	Ytest = to_categorical(Ytest, numOfClasses)
	Yvalidation = to_categorical(Yvalidation, numOfClasses)	
	return Ytrain, Ytest, Yvalidation, datagen

def myModel(datagen, Xtrain, Ytrain, Xvalidation, Yvalidation):
	model = keras.Sequential()
	model.add(Conv2D(100, (5, 5), input_shape=(28, 28, 1), activation='relu'))
	model.add(Conv2D(100, (4, 4), activation='relu'))
	model.add(Conv2D(100, (3, 3), activation='relu'))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((3, 3)))
	
	model.add(Dropout(0.5))
	model.add(Flatten())
	
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	
	model.add(Dense(numOfClasses, activation='softmax'))
	model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
	history = model.fit_generator(
				datagen.flow(Xtrain, Ytrain, batch_size=100),
				 epochs=5,
				validation_data=(Xvalidation, Yvalidation),
				shuffle=0)
	return history, model


def plotGraph(history):
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

def evaluate(model, Xtest, Ytest):
	score = model.evaluate(Xtest, Ytest, verbose=0)
	print('Test Score = ', score[0])
	print('accuracy Score = ', score[1])
	
	
image, classNumber, numOfClasses = DatasetImage()
Xtrain, Xtest, Ytrain, Ytest, Xvalidation, Yvalidation = splitingDataset(image)
Xtrain, Xtest, Xvalidation = DatasetToList(Xtrain, Xtest, Xvalidation)
Ytrain, Ytest, Yvalidation, datagen = changesIn_Y_Set(Xtrain, Ytrain, Ytest, Yvalidation, numOfClasses)
history, model = myModel(datagen, Xtrain, Ytrain, Xvalidation, Yvalidation)
plotGraph(history)
evaluate(model, Xtest, Ytest)

print("Prediction of Image")
r = 'Image Directory'
img1 = cv2.imread(r)
img1 = preprocessing(img1)
img1 = np.reshape(img1, [1, 28, 28, 1])

classIndex = int(model.predict_classes(img1))
print("class: ",classIndex)

prediction = model.predict(img1)
print("ppred : ", prediction)
probval = np.amax(prediction)
print(classIndex, probval)
