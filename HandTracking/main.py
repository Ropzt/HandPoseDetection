import cv2
import mediapipe as mp
import HandTrackingModule as htm

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1) #Mandatory : the network has been trained to recognize only one hand per image
mpDraw = mp.solutions.drawing_utils
stream = cv2.VideoCapture(0)                    #access cam

while True :
        ret, frame = stream.read()              #read cam and put the data in frame
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRGB)
        if results.multi_hand_landmarks :
            joint_list = []
            frame,joint_list = htm.track_hand(results,frame)
        cv2.imshow("HandTracking", frame)                #display frame
        if cv2.waitKey(1) & 0xFF == ord('q'):   #if 'Q' is pressed, Quit
            break
stream.release()
cv2.destroyAllWindows()