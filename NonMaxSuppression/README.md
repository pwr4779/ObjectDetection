# Non Maximum suppression

You Only Look Once is a real-time object detection algorithm, that avoids spending too much time on generating region proposals.Instead of locating objects perfectly, it prioritises speed and recognition.
This leads to a technique which filters the proposals based on some criteria called Non-maximum Suppression.  


 Non-Maximal Suppression (NMS) keeps the best bounding box. The first step in NMS is to remove all the predicted bounding boxes that have a detection probability that is less than a given NMS threshold. In the code below, we set this NMS threshold to 0.6. This means that all predicted bounding boxes that have a detection probability less than 0.6 will be removed.  


 After removing all the predicted bounding boxes that have a low detection probability, the second step in NMS, is to select the bounding boxes with the highest detection probability and eliminate all the bounding boxes whose Intersection Over Union (IOU) value is higher than a given IOU threshold. In the code below, we set this IOU threshold to 0.4. This means that all predicted bounding boxes that have an IOU value greater than 0.4 with respect to the best bounding boxes will be removed.  


Do NMS separate for eache class  
discarding all bouding boxes <code>&#60;</code> probability threshold
while BoundingBoxes:
- Take out the largest probability box
- Remove all other boxes with IoU > threshold

Do this for each class

### not using Non Maximum suppression 
<table border="0">
<tr>
    <td>
    <img src="https://github.com/pwr4779/ObjectDetection/blob/main/NonMaxSuppression/before.png"/>
    </td>
</tr>
</table>


### using Non Maximum suppression  
<table border="0">
<tr>
    <td>
    <img src="https://github.com/pwr4779/ObjectDetection/blob/main/NonMaxSuppression/after.png"/>
    </td>
</tr>
</table>

* [model: yolov3](https://pjreddie.com/darknet/yolo/)
<table border="0">
<tr>
    <td>
    <img src="https://github.com/pwr4779/ObjectDetection/blob/main/NonMaxSuppression/yolov3.png"/>
    </td>
</tr>
</table>

### iou code  
```python
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
```

### non Maximum suppression
```python
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
```


