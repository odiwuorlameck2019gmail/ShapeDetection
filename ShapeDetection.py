import cv2 
import numpy as np 

#SET the font type.

font=cv2.FONT_HERSHEY_COMPLEX
#Read image from source.
img=cv2.imread("shape_detection.png")

#Convert the image into grayscale.

grayimage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Deffine the image threshold.
ret,threshold=cv2.threshold(grayimage,239,256,cv2.THRESH_BINARY)

#get the contours .
contours,hierachy=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#Loop through all of the contours.

for contour in contours:
    approax=cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
    cv2.drawContours(img,[approax],0,(0),5)
    x=approax.ravel()[0]
    y=approax.ravel()[1]

    if len(approax)==3:
        cv2.putText(img,"Triangle",(x,y),font,1,(255,0,255),1)
    elif len(approax)==4:
        cv2.putText(img,"Rectangle",(x,y),font,1,(255,0,255),1)
    elif len(approax)==5:
        cv2.putText(img,"Pentagon",(x,y),font,1,(255,0,255),1)
    elif 6<len(approax)<15:
        cv2.putText(img,"Ellipse",(x,y),font,1,(255,0,255),1)
    else:
        cv2.putText(img,"Circle",(x,y),font,1,(255,0,255),1)
    
cv2.imshow("Shapes",img)
cv2.imshow("Threshold",threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()




