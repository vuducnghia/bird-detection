from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("results/weights/best.pt")

# detect from a image
# results = model.predict(source="test/images/000058.png", imgsz=640, conf=0.45)
# for r in results:
#     print(r.boxes.xywh)
#     print(r.boxes.conf)
#     print(r.speed)
#
#     # plot result:
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#     im.show()
#     im.save('results.jpg')  # save image


# detect from camera
cam = cv2.VideoCapture(0)
cv2.namedWindow("video")
img_counter = 0
while True:
    ret, frame = cam.read()
    img_counter += 1
    # print(img_counter)
    if img_counter % 3 == 0:
        continue
    results = model(frame, imgsz=640, conf=0.55)
    # print(results)
    for r in results:
        cv2.imshow("test", r.plot())

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
