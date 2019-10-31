import cv2

img = cv2.imread('krishana.jpg')

cv2.imshow('Happy Krishna Janmashtami', img)

cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image
