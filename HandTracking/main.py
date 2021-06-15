import cv2
import mediapipe as mp
import HandTrackingModule as htm
import tensorflow as tf
import RegNet as rg

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)  # Mandatory : the network has been trained to recognize only one hand per image
mpDraw = mp.solutions.drawing_utils
stream = cv2.VideoCapture(0)                    # access cam

regnet = rg.RegNet34()
optimizer = tf.keras.optimizers.Adadelta(lr=1e-4)

# We need to compile the model at the begining
# possibility to load an existing model but custom layer error is still an unresolved issue to this day
regnet.model.compile(optimizer=optimizer,
                      loss=['mse', 'mse', 'mse'],
                      loss_weights=[100, 100, 1],
                      metrics=['mse'])
regnet.model.load_weights('/weights/weights.h5')

while True :
        ret, frame = stream.read()              # read cam and put the data in frame
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRGB)
        if results.multi_hand_landmarks:
            frame, joint_list = htm.track_hand(results, frame, regnet)
        cv2.imshow("HandTracking", frame)                # display frame
        if cv2.waitKey(1) & 0xFF == ord('q'):   # if 'Q' is pressed, Quit
            break
stream.release()
cv2.destroyAllWindows()
