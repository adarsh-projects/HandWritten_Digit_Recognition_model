from infix_To_Postfix import Itop
from postfix_Evaluation import pe
from CNN_model import predictImage, model
import cv2
import os
import numpy as np
path = '//home//adarsh//Desktop//check//image'

interger_mapping = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '(', 11: ')', 12: '+', 13: '-', 14: '*', 15: '/'}

pl = predictImage()

file1 = os.listdir(path)
file1.sort() # sort all the file in order

str1 = ""

for img in file1:
	
	p = path+'//'+img
	img = cv2.imread(p)
	img = cv2.resize(img, (28,28))
	img = pl.preprocessing(img)
	img1 = np.reshape(img, [1, 28, 28, 1])
	
	classIndex = int(model.predict_classes(img1))
	
	str1 = str1 + interger_mapping[classIndex]

itop = Itop.infixtopostfix(str1)
print(itop)
print(pe.solve_postfix_expression(itop))
