from ultralytics import YOLO

model = YOLO('yolov8x.pt')

result = model.predict('input_videos/Untitled design.mp4', save=True, project='output', name='test_run')


print(result)

print("Boxes:")
for box in result[0].boxes:
    print(box)

