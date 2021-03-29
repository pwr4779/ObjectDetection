import cv2
import os.path as path
import cv2
import numpy as np
import torch
from typing import List
from util import non_max_suppression, intersection_over_union

classes = ["person", "bicycle", "car", "motorcycle",
           "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
           "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
           "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
           "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
           "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
           "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
           "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
           "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
           "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]




if __name__ == '__main__':

    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # 클래스의 갯수만큼 랜덤 RGB 배열을 생성
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    video_file = 'test_video.mp4'
    out_path = './out_frames'

    cap = cv2.VideoCapture(video_file)

    counter = 0
    while True:

        ret, frame = cap.read()

        if ret:
            size = 320
            height, width, channels = frame.shape

            blob = cv2.dnn.blobFromImage(frame, 0.00392, (size, size), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            bboxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.7:
                        # 탐지된 객체의 너비, 높이 및 중앙 좌표값 찾기
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # 객체의 사각형 테두리 중 좌상단 좌표값 찾기
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        bboxes.append([class_id, confidence, x, y, x + w, y + h])
            # 후보 박스(x, y, width, height)와 confidence(상자가 물체일 확률) 출력

            # using nonMaxSuppression
            bboxes = non_max_suppression(bboxes, 0.8, 0.6)
            # 후보 박스(x, y, width, height)와 confidence(상자가 물체일 확q률) 출력

            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            for i in range(len(bboxes)):
                x1, y1, x2, y2 = bboxes[i][2:]
                label = f'{str(classes[bboxes[i][0]])} {float(bboxes[i][1]):.2f}'
                color = colors[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 + 30), font, 0.7, color, 1)

            # cv2.imwrite(path.join(out_path, '{:06d}.jpg'.format(counter)), frame)
            cv2.imshow('', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            counter += 1


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    exit()
