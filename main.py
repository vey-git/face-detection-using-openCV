import cv2


"""
cascading classifier -> differentiates different objects that you want to track
eye cascade
face cascade
gesture cascade 

parse in a xml file within the classifier with data to represent that classifier 



"""

#parse in a xml file within the classifier with data to represent that classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#test with images
image = cv2.imread("image1.png")





