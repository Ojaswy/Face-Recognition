import cv2
import sys

# Command-line arguments
imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# initialize the haar-cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    #Location of image from the eye
    scaleFactor=1.2,
    #Neighbouring faces - only for accuracy
    minNeighbors=7,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)
print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
