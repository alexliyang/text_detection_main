import cv2
img = cv2.imread("yi.jpg")
cv2.imshow("guoyi", img)
imgScale = cv2.resize(img,None,None,fx=0.5,fy=0.5)
cv2.imshow("g",imgScale)
cv2.waitKey(0)
