import cv2
import numpy as np
import torch
from typing import List


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

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# 클래스의 갯수만큼 랜덤 RGB 배열을 생성
colors = np.random.uniform(0, 255, size=(len(classes), 3))


img = cv2.imread("test.jpg")
size = 416
height, width, channels = img.shape

# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

blob = cv2.dnn.blobFromImage(img, 0.00392, (size, size), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []
bboxes = []
"""
output : N x N x (5 + C) x B)
N * N : Grid
C : # of Classes
5 : Bounding Box info (x, y, w, h, confidence)
B : Grid cell 당 박스 수
"""
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.1:
            # 탐지된 객체의 너비, 높이 및 중앙 좌표값 찾기
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # 객체의 사각형 테두리 중 좌상단 좌표값 찾기
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            bboxes.append([class_id, confidence, x, y, x+w, y+h])
# 후보 박스(x, y, width, height)와 confidence(상자가 물체일 확률) 출력
print(f"boxes: {boxes}")
print(f"confidences: {confidences}")
print(f"bboxes: {bboxes}")
# not using nonMaxSuppression
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
for i in range(len(boxes)):
    x, y, w, h = boxes[i]
    label = f'{str(classes[class_ids[i]])} {float(confidences[i]):.2f}'
    color = colors[i]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x, y + 30), font, 0.7, color, 1)
cv2.imshow("Image", img)
cv2.imwrite('before.png', img)


def intersection_over_union(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou


def non_max_suppression(
        bboxes,
        prob_threshold,
        iou_threshold,
):
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    nms_bboxes = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
            )< iou_threshold
        ]
        nms_bboxes.append(chosen_box)
    return nms_bboxes

img = cv2.imread("test.jpg")
size = 416
height, width, channels = img.shape

# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

blob = cv2.dnn.blobFromImage(img, 0.00392, (size, size), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# using nonMaxSuppression
bboxes = non_max_suppression(bboxes, 0.7, 0.6)
# 후보 박스(x, y, width, height)와 confidence(상자가 물체일 확률) 출력
print(f"bboxes: {bboxes}")
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
for i in range(len(bboxes)):
    x1, y1, x2, y2 = bboxes[i][2:]
    label = f'{str(classes[bboxes[i][0]])} {float(bboxes[i][1]):.2f}'
    color = colors[i]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1 + 30), font, 0.7, color, 1)
cv2.imshow("nms", img)
cv2.imwrite('after.png', img)
cv2.waitKey()
cv2.destroyAllWindows()