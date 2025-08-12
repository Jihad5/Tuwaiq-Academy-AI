from djitellopy import tello 
import cv2
from ultralytics import YOLO

# Load a model
model = YOLO("best.onnx")
me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()

while True:
    img = me-get_frame_read().frame
    img = cv2.cvtColor (img, cv2.COLOR_BGR2RGB)
    img = cv2. resize(img, (640, 480))
    output = model.predict(img, stream=False, show=True)
    # classes = output [0] • names
    # boxes = output [0]. boxes. xyxyl
    # for i in range(boxes.shape[®️]):
    #img = cv2. rectangle(img=img,
    #pt1= (int (boxes [1,0]), int (boxes [1,1])),
    #pt2=(int (boxes[1,2]), int(boxes[1,3])),
    #color=(0,255,0),
    #thickness=2)
    #img = cv2. putText(img,
    #classes [int(output [0]. boxes.cls[1])], (int (boxes [i,0]), int (boxes [1,1])), CV2. FONT_HERSHEY_SIMPLEX, 1, (255,255,0),2)
    # cv2. imshow "Image", img)
    if cv2.waitkey(5) & OxFF == ord('q'):
        me.streamoff()
        break
cv2.destroyA11Windows()