# from ultralytics import YOLO
# from ultralytics.yolo.v8.detect.predict import DetectionPredictor
# import cv2
# import numpy as np

# model = YOLO("phantom-segmentation.pt")
# results = model.predict(source=0, show=True)

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from ultralytics.yolo.utils.plotting import Annotator

model = YOLO("phantom-segmentation.pt")
model.to("cuda")

# Shoelace Formula in cm^2
def scale_and_transform(vertices):
    scaled_vertices = []
    for x, y in vertices:
        scaled_x = (355 - x) * (81.3 / 480)
        scaled_y = (y - 338) * (115 / 640)
        scaled_vertices.append([scaled_x, scaled_y])
    return scaled_vertices

def shoelace_area(vertices):
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += (vertices[i][0] * vertices[j][1]) - (vertices[j][0] * vertices[i][1])
    return abs(area) / 2.0

# # # Load the YOLOv8 model
def YOLODetect(img):
    results = model(
        img,
        conf=0.6,
        device=0,
        show=False
    )
    # annotated_frame = results[0].plot()
    # print(results[0])
    # print(results[0].masks)
    h = img.shape[0]
    w = img.shape[1]
    
    img = np.zeros(img.shape, dtype=np.uint8)
    res_boxes = []
    point_segment = []
    boxes = results[0].boxes.cpu().numpy()
    if boxes:
        for index, box in enumerate(boxes):
            x = (box.xywh[0][0]).astype(int)
            y = (box.xywh[0][1]).astype(int)
            name = box.cls[0].astype(int)

            res_boxes.append([x, y, name])

    masks = results[0].masks
    color= (255, 255, 255)
    # cv2.rectangle(img, (235, 134), (437, 367), (255, 255, 255), thickness=10)
    if masks:
        for index, mask in enumerate(masks):
            x = res_boxes[index][0]
            y = res_boxes[index][1]
            c = res_boxes[index][2]
            # print(x,y,c)
            if (x > 240 and x < 437) and (y > 120 and y < 350):
                    
                if c == 0:
                    color=(255,0,0)
                    points = np.array(mask.xy, np.int32)
                    points_new = np.squeeze(points)
                    # print(points_new)
                    scaled_points = scale_and_transform(points_new)
                    area_value = shoelace_area(scaled_points)
                    # sorted_indices = np.argsort(points_new[:, 0])
                    # sorted_points_new = points_new[sorted_indices]
                    # sorted_indices_y = np.lexsort((sorted_points_new[:, 1], sorted_points_new[:, 0]))
                    # sorted_points_new = sorted_points_new[sorted_indices_y]
                    # filtered_points = []
                    # x_prev = sorted_points_new[0][0]
                    # y_min, y_max = sorted_points_new[0][1], sorted_points_new[0][1]

                    # for i in range(1, len(sorted_points_new)):
                    #     x_current = sorted_points_new[i][0]
                    #     y_current = sorted_points_new[i][1]
                        
                    #     if x_current == x_prev:
                    #         y_min = min(y_min, y_current)
                    #         y_max = max(y_max, y_current)
                    #     else:
                    #         filtered_points.append([x_prev, y_min])
                    #         x_prev = x_current
                    #         y_min, y_max = y_current, y_current

                    # Menambahkan pasangan nilai terakhir
                    # filtered_points.append([x_prev, y_min])

                    # filtered_points = np.array(filtered_points)

                    # print(filtered_points)
                elif c == 1:
                    color=(0,0,255)
                    points = np.array(mask.xy, np.int32)
                    points_new = np.squeeze(points)
                    x_values = [point[0] for point in points_new]
                    y_values = [point[1] for point in points_new]
                    index_x_min = x_values.index(min(x_values))
                    index_x_max = x_values.index(max(x_values))
                    point_segment.append([x_values[index_x_min], y_values[index_x_min], x_values[index_x_max], y_values[index_x_max]])

                    scaled_points = scale_and_transform(points_new)
                    area_value = shoelace_area(scaled_points)
                else:
                    color=(0,0,0)
            # print("Coordinates:", x, ":", y, "Object:", c)
            # points = np.array(mask.xy, np.int32)
            # points_new = np.squeeze(points)
            # scaled_points = scale_and_transform(points_new)
            # area_value = shoelace_area(scaled_points)
            # print(area_value)
            # print(points)
            # point_segment.append
            points = points.reshape((-1, 1, 2))
            overlay = img.copy()
            # print(point_segment)
            cv2.fillPoly(img, [points], color )
            img = cv2.addWeighted(overlay, 0.5, img, 1, 0)

    return img,res_boxes,point_segment


# import cv2
# from ultralytics import YOLO

# Load the YOLOv8 model
# model = YOLO("yolov8l-seg.pt")

# # Open the video file
# # video_path = "path/to/your/video/file.mp4"
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4,480)

# # # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()
#     height, width = frame.shape[:2]
#     mask = np.zeros((height, width), dtype=np.uint8)
#     x_range = (235, 437)
#     y_range = (134, 367)
#     if success:
#         # Run YOLOv8 inference on the frame
#         # annotated_frame = YOLODetect(frame)
#         for y in range(height):
#             for x in range(width):
#                 if x < x_range[0] or x > x_range[1] or y < y_range[0] or y > y_range[1]:
#                     mask[y, x] = 0  # Set to black
#                 else:
#                     mask[y, x] = 255  # Set to white

#         #     boxes = r.boxes
#         #     for box in boxes:
#         #         b = box.xyxy[
#         #             0
#         #         ]  # get box coordinates in (top, left, bottom, right) format
#         #         c = box.cls
#         #         annotator.box_label(b, model.names[int(c)])

#         # frame = annotator.result()

#         # Display the annotated frame
#         darkened_image = cv2.bitwise_and(frame, frame, mask=mask)
#         # darkened_image = YOLODetect(darkened_image)
#         cv2.imshow("YOLOv8 Inference", darkened_image)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()
