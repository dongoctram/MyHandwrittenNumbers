import numpy as np
import matplotlib.pyplot as plt

# Path to your saved data
save_dir = 'D:\\AIprojetcs\\HandwrittenNumbers\\saveimages'

# Load images and labels
images = np.load(f'{save_dir}\\images.npy')
labels = np.load(f'{save_dir}\\labels.npy')

# Display first few images
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap='gray')
    ax.axis('off')
    ax.set_title(f'Label: {labels[i]}')
plt.tight_layout()
plt.show()
