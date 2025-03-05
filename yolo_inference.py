from ultralytics import YOLO

model = YOLO('yolov8x.pt')

result = model.track('/Users/patricklu/Documents/GitHub/Smashlytics/input_videos/CondenseTrainingSpeed.mp4', conf=0.2, save=True, project='output', name='test_run')


print(result)

print("Boxes:")
for box in result[0].boxes:
    print(box)

