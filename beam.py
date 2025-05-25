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

# Crack Type Classification
crack_type_classes = ['Diagonal', 'Horizontal', 'Vertical', 'Web_and_X']
crack_type_model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
crack_type_model.fc = nn.Sequential(
    nn.Linear(crack_type_model.fc.in_features, len(crack_type_classes)),
    nn.Sigmoid()
)
# Crack Size Classification
crack_size_classes = ['Big', 'Small']
crack_size_model = torchvision.models.shufflenet_v2_x1_5(weights='IMAGENET1K_V1')
crack_size_model.fc = nn.Sequential(
    nn.Linear(crack_size_model.fc.in_features, len(crack_size_classes)),
    nn.Sigmoid()
)
# Device and model loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
crack_type_model.load_state_dict(torch.load('./crack_classification/full/full_crack_type_best_model.pth', map_location=device))
crack_size_model.load_state_dict(torch.load('./crack_classification/full/full_crack_size_best_model.pth', map_location=device))
crack_type_model.to(device).eval()
crack_size_model.to(device).eval()

imagenet_stats = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
transform = transforms.Compose([
    transforms.CenterCrop((400, 400)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_stats[0], imagenet_stats[1])
])
crack_criteria = {
    ('Diagonal', 'Big'): 4,
    ('Diagonal', 'Small'): 5,
    ('Horizontal', 'Big'): 8,
    ('Horizontal', 'Small'): 9,
    ('Vertical', 'Big'): 6,
    ('Vertical', 'Small'): 7,
    ('Web_and_X', None): 3
}
criteria_map = {
    18: [0],
    19: [3, 4, 6, 8],
    20: [1, 5, 7, 9, 10]
}
def combine_detection_classification_results(detections, crack_class_pairs):
    criteria_set = set()
    if 'Exposed rebar' in detections:
        criteria_set.add(0)
    if len(detections) == 0:
        criteria_set.add(1)

    for crack_type, crack_size in crack_class_pairs:
        key = (crack_type, crack_size) if crack_type != 'Web_and_X' else ('Web_and_X', None)
        if key in crack_criteria:
            criteria_set.add(crack_criteria[key])

    if any(c in criteria_set for c in criteria_map[18]):
        damage_class = 18
    elif any(c in criteria_set for c in criteria_map[19]):
        damage_class = 19
    else:
        damage_class = 20

    valid_criteria = [c for c in criteria_set if c in criteria_map[damage_class]]

    return damage_class, sorted(valid_criteria)



model = YOLO('./damage_detection/yolov9_full/weights/best.pt') 

path = './test_data/beam'
img_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.png'))],
                   key=natural_sort_key)
results = model.predict(source=img_files, save=True, conf=0.25)

ans = []

for i, result in enumerate(results):
    image_path = img_files[i]
    detections = result.names
    labels = [result.names[int(cls)] for cls in result.boxes.cls.cpu().numpy()]

    detected_crack_pairs = []
    if 'Crack' in labels and 'Exposed rebar' not in labels:
        image = Image.open(image_path).convert('RGB')

        for j, (box, cls_id) in enumerate(zip(result.boxes.xyxy, result.boxes.cls)):
            class_name = result.names[int(cls_id)]
            x1, y1, x2, y2 = map(int, box.tolist())
            cropped = image.crop((x1, y1, x2, y2))
            crop_filename = f'./cropped_cracks/{i}_crop{j}.jpg'
            cropped.save(crop_filename)
            image_tensor = transform(cropped).unsqueeze(0).to(device)
            with torch.no_grad():
                # Type prediction
                type_output = crack_type_model(image_tensor)
                type_pred = type_output.argmax(dim=1).item()
                crack_type = crack_type_classes[type_pred]
                crack_size = None
                if crack_type in ['Diagonal', 'Horizontal', 'Vertical']:
                    # Size prediction
                    size_output = crack_size_model(image_tensor)
                    size_pred = size_output.argmax(dim=1).item()
                    crack_size = crack_size_classes[size_pred]
                detected_crack_pairs.append((crack_type, crack_size))
    damage_class, criteria = combine_detection_classification_results(labels, detected_crack_pairs)
    class_str = ','.join(map(str, [damage_class] + criteria))
    ans.append([i+1, class_str])

with open('./test_data/beam.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'class'])
    writer.writerows(ans)
