



import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt

interpreter = tf.lite.Interpreter(model_path="C:\\Users\\namru\\Documents\\Pose\\Movemodel\\movenet-tflite-singlepose-lightning-v1\\3.tflite")
interpreter.allocate_tensors()

 
def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)
    cv2.putText(img, "Tracking", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, _ = frame.shape
    for kp in keypoints:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 2, (0, 255, 0), -1)

EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c', (0, 5): 'm', (0, 6): 'c',
    (5, 7): 'm', (7, 9): 'm', (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm', (12, 14): 'c', (14, 16): 'c'
}

def draw_connections(frame, keypoints, edges, confidence_threshold):
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

cap = cv2.VideoCapture(r"C:\Users\namru\Documents\Pose\SAMout.mp4")
tracker = cv2.TrackerCSRT_create()

output_width = 848
output_height = 480

success, img = cap.read()
if not success:
    print("Error: Cannot read video file")
else:
    bbox = cv2.selectROI("Tracking", img, False)
    tracker.init(img, bbox)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        timer = cv2.getTickCount()
        
        success, bbox = tracker.update(frame)
        
        if success:
            drawBox(frame, bbox)
            x, y, w, h = [int(v) for v in bbox]
            roi = frame[y:y+h, x:x+w]

            
            img_resized = tf.image.resize_with_pad(tf.expand_dims(roi, axis=0), 192, 192)
            input_img = tf.cast(img_resized, dtype=tf.float32)

            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            interpreter.set_tensor(input_details[0]['index'], np.array(input_img))
            interpreter.invoke()

           
            keypoints_with_scores = interpreter.get_tensor(output_details[0]['index']).reshape((1, 17, 3))

            
            keypoints_with_scores_rescaled = []
            for keypoint in keypoints_with_scores[0]:
                ky, kx, kp_conf = keypoint
                keypoints_with_scores_rescaled.append([ky * h + y, kx * w + x, kp_conf])
            
            keypoints_with_scores_rescaled = np.array(keypoints_with_scores_rescaled)

            
            loop_through_people(frame, [keypoints_with_scores_rescaled], EDGES, 0.3)
        else:
            cv2.putText(frame, "Lost", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        
        cv2.imshow('Movenet Singlepose', frame)
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()







