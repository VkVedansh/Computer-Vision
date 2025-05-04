from deepface import DeepFace
import cv2

#load the pre trained face detection

face_Cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


#open Camera
cap = cv2.VideoCapture(0)# 0 fro internal camera,1 for web cam
while True:
    ret,frame = cap.read()
    if not ret:
        break

    #convert frame to graeyscale from face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #dectect face
    faces = face_Cascade.detectMultiScale(gray_frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))#minneighbors-> no of face can be detected

    for(x,y,w,h) in faces:
        #crop face region
        face_rio = frame[y:y+h,x:x+w]

        try: # documutentation
            #analysis the face emotions
            analysis = DeepFace.analyze(face_rio,actions=['emotion'],enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
        except:
            emotion = 'UnNone'


        #draw rectangle around face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame,emotion,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)

    
    #show frames
    cv2.imshow("Real-Time Emotion Detection",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()