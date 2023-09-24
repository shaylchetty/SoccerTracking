def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    :param box1: Tuple (x1, y1, x2, y2) representing the first box.
    :param box2: Tuple (x1, y1, x2, y2) representing the second box.
    :return: IoU score.
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate the coordinates of the intersection rectangle
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    # Calculate the area of the intersection
    intersection_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calculate the area of both boxes
    area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Calculate the Union (sum of both box areas - intersection)
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


def shayl_non_max_suppression(boxes, threshold=0.5):
    """
    Apply non-maximum suppression to a list of bounding boxes.
    :param boxes: List of bounding boxes, where each box is represented as (x1, y1, x2, y2).
    :param threshold: IoU threshold to determine duplicates.
    :return: List of filtered bounding boxes.
    """
    if len(boxes) == 0:
        return []

    # Sort the boxes by their bottom-right y-coordinate (y2)
    sorted_boxes = sorted(boxes, key=lambda x: x[3], reverse=True)

    # Initialize a list to store the selected boxes
    selected_boxes = [sorted_boxes[0]]

    # Iterate through the sorted boxes
    for box in sorted_boxes[1:]:
        iou_scores = [calculate_iou(box, sel_box) for sel_box in selected_boxes]

        if all(iou < threshold for iou in iou_scores):
            selected_boxes.append(box)

    return selected_boxes
