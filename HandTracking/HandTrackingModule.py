import cv2
import numpy as np

def track_hand(results,frame, net) :
    """

    :param results:
    :param frame:
    :param net:
    :return:
    """
    handLms = results.multi_hand_landmarks[0]
    xList=[]
    yList=[]
    for id, lm in enumerate(handLms.landmark) :
        h,w,c = frame.shape
        cx,cy = int(lm.x*w), int(lm.y*h)
        xList.append(cx)
        yList.append(cy)
    xmin, xmax = min(xList), max(xList)
    ymin, ymax = min(yList), max(yList)
    xmin -= 10
    xmax += 10
    ymin -= 10
    ymax += 10
    bxmin = xmin
    bymin = ymin
    if ((xmax-xmin)>(ymax-ymin)) :
        bxmax = xmax
        bymax = ymin + (xmax-xmin)
    else :
        bymax = ymax
        bxmax = xmin + (ymax - ymin)
    boundbox = bxmin,bymin,bxmax,bymax

    crop_img = frame[boundbox[1]:boundbox[3], boundbox[0]:boundbox[2]].copy()
    final_frame = cv2.resize(crop_img, (256,256), interpolation=cv2.INTER_CUBIC)
    final_frame = np.reshape(final_frame, (1,256,256,3))
    result = net.model.predict_on_batch(final_frame)
    heatmap = result[2]
    joint_list = []
    pos_list = []
    max_val = 0
    for i in range(0, 21):
        for j in range(0, 32):
            for k in range(0, 32):
                if heatmap[0][j][k][i] > max_val:
                    max_val = heatmap[0][j][k][i]
                    max_j = j
                    max_k = k
        pos_list.append(max_j)
        pos_list.append(max_k)
        joint_list.append(np.asarray(pos_list))
        pos_list = []
        max_j = 0
        max_k = 0
        max_val = 0
    joint_list = np.asarray(joint_list)
    joint_list = joint_list * 8
    for i in range(0,21) :
        joint_list[i][0] += bxmin
        joint_list[i][1] += bymin
        cv2.circle(frame, joint_list[i], radius=2, color=(255, 0, 255))
    return frame,joint_list
