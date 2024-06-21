import cv2
import numpy as np
import os

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0  # Normalize to [0, 1]
    return img

def load_dataset(data_dir, save_dir):
    images = []
    labels = []
    for label in range(10):
        label_dir = os.path.join(data_dir, str(label))
        for image_file in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_file)
            img = preprocess_image(image_path)
            images.append(img)
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    
    # Save preprocessed data to the specified directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    np.save(os.path.join(save_dir, 'images.npy'), images)
    np.save(os.path.join(save_dir, 'labels.npy'), labels)
    
    return images, labels

# Example usage
data_dir = 'D:\\AIprojetcs\\HandwrittenNumbers\\sourceimages'
save_dir = 'D:\\AIprojetcs\\HandwrittenNumbers\\saveimages'
images, labels = load_dataset(data_dir, save_dir)

# Verify the dataset shape
print(f'Images shape: {images.shape}')
print(f'Labels shape: {labels.shape}')
