import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i,","--image",required=True,help="Path to input image")
args = vars(ap.parse_args())


image = cv2.imread(args["image"]) #read image
print(image)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #convert image to gray

ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY) #binarize the image
imh, cnts, heir = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 

#find shapes in the image
orig = image.copy()

#contoured = cv2.drawContours(orig,cnts,-1,(255,0,0),3) #draw the contours on the image

# Sort the contours according to the contour area
if len(cnts) > 1:
    cnts_sorted = sorted(cnts,key=cv2.contourArea,reverse=True)[:-1]
    cnts_sorted


r = cnts_sorted[0] # get the largest contour which should be the paper

# generate an approximate shape for the rectangle

sss = cv2.arcLength(r,True)
approx = cv2.approxPolyDP(r,0.02*sss,True)



print("Approx ",approx)


top_left = approx[1]
top_right = approx[0]
bottom_left = approx[2]
bottom_right = approx[3]


rect = [top_left,top_right,bottom_left,bottom_right]
rect = np.array(rect,dtype="float32")
dst = [[0,0],[400,0],[0,500],[400,500]]
dst = np.array(dst,dtype="float32")


maxWidth = 400
maxHeight = 500

print("hi there")

m = cv2.getPerspectiveTransform(rect,dst)
warped = cv2.warpPerspective(image, m, (maxWidth, maxHeight))

print("dont make me laugh")


image = cv2.resize(image,(400,500))
cv2.imshow("warped",warped)
cv2.imshow("original image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()