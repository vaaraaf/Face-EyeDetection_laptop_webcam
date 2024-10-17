import cv2
from PIL import Image
face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
camera = cv2.VideoCapture(0)
# sample_image = cv2.imread('./images/sample.jpg')
# sample_image_gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = camera.read()
    # cv2.imshow('webcam', frame)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detection.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))
    print(len(faces))
    for (x,y,w,h) in faces:
        print(x,y,x+w, y+h)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 4)

    eyes = eye_detection.detectMultiScale(frame_gray, minNeighbors=9)
    for (ex, ey, ew, eh) in eyes:
         cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0,0,255), 2)
    cv2.imshow('webcam', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# cv2.imshow('Faces', sample_image)
camera.release()
cv2.destroyAllWindows()