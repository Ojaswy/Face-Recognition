import cv2                                          
import sys                                          

# Uncomment the next line to enter image path as a Command-line arguments
#imagePath = sys.argv[1]
imagePath = 'coldplay.jpg'                            
cascPath = "haarcascade_frontalface_default.xml"    

faceCascade = cv2.CascadeClassifier(cascPath)       

# Read the image
image = cv2.imread(imagePath)                      
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     

# Detect faces in the image
faces = faceCascade.detectMultiScale(               
    gray,                                           
    scaleFactor=1.1,                                
    minNeighbors=3,                                 
    minSize=(30, 30)                                                                      
)

print("I have Found {0} faces!".format(len(faces)))        


# Draw a rectangle around the faces                 
for (x, y, w, h) in faces:                                       
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)     

cv2.imshow("Faces found", image)                    
cv2.waitKey(0)                                      
