import cv2
import sys 
import datetime as dt

cascPath = "haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

print("\n\nDetection Starts at :  {0}  \n\n".format(str(dt.datetime.now())))

while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.waitKey(100)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=0
    )
    # Draw a rectangle around the faces

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('Video', frame)
    print("Found {0} faces , at {1} ".format(len(faces),str(dt.datetime.now())))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything is done, press q to exit.
print("\n\nDetection Ends at :  {0}  \n\n\n".format(str(dt.datetime.now())))
video_capture.release()
cv2.destroyAllWindows()
