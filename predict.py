from ultralytics import YOLO
from PIL import Image

model = YOLO("results/weights/best.pt")

results = model.predict(source="test/images/000058.png", imgsz=640, conf=0.45)
for r in results:
    print(r.boxes.xywh)
    print(r.boxes.conf)
    print(r.speed)

    # plot result:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()
    im.save('results.jpg')  # save image
