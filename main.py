import cv2


"""
cascading classifier -> differentiates different objects that you want to track
eye cascade
face cascade
gesture cascade 

parse in a xml file within the classifier with data to represent that classifier 



"""

#parse in a xml file within the classifier with data to represent that classifier
"""face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#test with images
image = cv2.imread("image1.png")

#convert image to black and white
grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#detect the faces, takes in 3 parameters, the image greyscaled, a scale factor (how much it is cropped by each time to see if its a face
# and a minimum neighbour -> if it detects 4 things that look like a face but min is 5 then it wont declare it as a face.
faces = face_cascade.detectMultiScale(grey_image, 1.2, 3)



for (x, y, width, height) in faces:
    #rectangle to create a rectangle outline of the face
    cv2.rectangle(image, (x,y), (x+width, y+height), (0,0,255), 6) #3 parameters the image, and the coordinates top left to bottom right and finally a colour in argb

#parse in window name (whatever you want it to be) and the image
cv2.imshow("Faces", image)
cv2.waitKey() #waits until we enter a key before displaying."""




#detecting faces from a live webcam

#parse in a xml file within the classifier with data to represent that classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

webcam = cv2.VideoCapture(0) #default webcam

while True:
    success, img = webcam.read()
    grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey_image, 1.2, 7)
    for (x, y, width, height) in faces:
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 6)
        cv2.imshow("Faces", img)
        cv2.waitKey()

