from ultralytics import YOLO
from collections import Counter
import pandas as pd
import csv
import os
import re
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('(\d+)', s)]
# Crack Classification
crack_classes = ['Diagonal','Horizontal','Horizontal_large','Vertical','Vertical_large','Web','X-shape']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
crack_model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
crack_model.fc = nn.Sequential(
    nn.Linear(crack_model.fc.in_features, len(crack_classes)),
    nn.Sigmoid()
)
if torch.cuda.is_available():
    crack_model.cuda()
crack_model.load_state_dict(torch.load('./crack_classification/column/model.pth', map_location=device))
crack_model.eval()
imagenet_stats = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
transform = transforms.Compose([
    transforms.CenterCrop((400, 400)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_stats[0], imagenet_stats[1])
])
crack_criteria = {
    'Diagonal': 4,
    'Horizontal': 9,
    'Horizontal_large': 8,
    'Vertical': 7,
    'Vertical_large': 6,
    'Web': 3,
    'X-shape': 3
}
criteria_map = {
    18: [0, 3, 4, 6, 8],
    19: [7],
    20: [1, 9]
}
def combine_detection_classification_results(detections, crack_class=None):
    criteria_set = set()
    if 'Exposed rebar' in detections:
        criteria_set.add(0)
    if len(detections)==0:
        criteria_set.add(1)

    if crack_class:
        for crack in crack_class:
            if crack in crack_criteria:
                criteria_set.add(crack_criteria[crack])
    
    if any(c in criteria_set for c in criteria_map[18]):
        return 18, sorted(criteria_set)
    elif any(c in criteria_set for c in criteria_map[19]):
        return 19, sorted(criteria_set)
    else:
        return 20, sorted(criteria_set)



model = YOLO('./damage_detection/yolov9_full/weights/best.pt') 

path = './test_data/column'
img_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.png'))],
                   key=natural_sort_key)
results = model.predict(source=img_files, save=True, conf=0.25)

ans = []

for i, result in enumerate(results):
    image_path = img_files[i]
    detections = result.names
    labels = [result.names[int(cls)] for cls in result.boxes.cls.cpu().numpy()]
    # print(i, labels)

    detected_cracks = []
    if 'Crack' in labels or 'Spalling' in labels:
        image = Image.open(image_path).convert('RGB')

        for j, (box, cls_id) in enumerate(zip(result.boxes.xyxy, result.boxes.cls)):
            class_name = result.names[int(cls_id)]
            if class_name != 'Crack' and class_name != 'Spalling':
                continue
            else:
                x1, y1, x2, y2 = map(int, box.tolist())
                cropped = image.crop((x1, y1, x2, y2))
                crop_filename = f'./cropped_cracks/{i}_crop{j}.jpg'
                cropped.save(crop_filename)
                image_tensor = transform(cropped).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = crack_model(image_tensor)
                    pred = output.argmax(dim=1).item()
                    predicted_label = crack_classes[pred]
                    detected_cracks.append(predicted_label)
    damage_class, criteria = combine_detection_classification_results(labels, list(set(detected_cracks)))
    class_str = ','.join(map(str, [damage_class] + criteria))
    ans.append([i+1, class_str])

with open('./test_data/column.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'class'])
    writer.writerows(ans)
