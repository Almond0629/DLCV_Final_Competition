from ultralytics import YOLO

model = YOLO('./damage_detection/yolov9_full/weights/best.pt')

if __name__ == '__main__':
    model.train(
        data='./damage_detection/3class/data.yaml',
        imgsz=640,
        epochs=85,                
        batch=16,                 
        device=0,                  # 0 for GPU, 'cpu' for CPU
        optimizer='AdamW',           
        lr0=0.01,                  
        weight_decay=0.0005,       
        project='damage_detection', # where to save runs
        name='yolov9_full',         # experiment name
        pretrained=True             
    )