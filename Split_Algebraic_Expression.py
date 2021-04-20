from PIL import Image
import os
import cv2

src = '//home//adarsh//Desktop//check//40.png' # Image file
im = Image.open(src)
k = 0
left = 130
right = 390
imgHeight = im.size[1] #560

for i in range(0, 5):
	box = (left, 100, right, imgHeight-40) #(left, top, right, bottom)
	a = im.crop(box)
	des = "//home//adarsh//Desktop//check//image//IMG-%"+str(k)+".png"
	a.save(des)
	img = cv2.imread(des)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	(thresh, img) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
	img1 = Image.fromarray(img)
	img1.save(des)
	k += 1
	left = right
	right += 260
