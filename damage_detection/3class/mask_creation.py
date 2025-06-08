from PIL import Image, ImageDraw
import os

def create_crack_mask_only(label_file, output_mask_file, image_size):
    width, height = image_size
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    crack_found = False
    with open(label_file, 'r') as f:
        for line in f:
            cls, x_center, y_center, w, h = map(float, line.strip().split())
            cls = int(cls)
            if cls != 1:
                continue

            crack_found = True
            x1 = int((x_center - w / 2) * width)
            y1 = int((y_center - h / 2) * height)
            x2 = int((x_center + w / 2) * width)
            y2 = int((y_center + h / 2) * height)
            draw.rectangle([x1, y1, x2, y2], fill=255)

    if crack_found:
        mask.save(output_mask_file)
        return True
    else:
        return False



images_dir = './damage_detection/3class/for_training/datasets_3class_full_shuffled/train/images'
labels_dir = './damage_detection/3class/for_training/datasets_3class_full_shuffled/train/labels'
masks_dir = './damage_detection/3class/masks'

os.makedirs(masks_dir, exist_ok=True)

for filename in os.listdir(labels_dir):
    if not filename.endswith('.txt'):
        continue
    label_path = os.path.join(labels_dir, filename)
    image_path = os.path.join(images_dir, filename.replace('.txt', '.jpg'))
    mask_path = os.path.join(masks_dir, filename.replace('.txt', '.png'))

    if not os.path.exists(image_path):
        continue

    with Image.open(image_path) as img:
        w, h = img.size
        has_crack = create_crack_mask_only(label_path, mask_path, (w, h))
        # if not has_crack:
        #     print(f"Skipping: {filename} (no cracks)")

