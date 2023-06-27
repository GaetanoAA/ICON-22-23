import os
import random
import imageio.v2 as imageio


class ImageLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    @staticmethod
    def load_image(file_path):
        image = imageio.imread(file_path)
        return image

    def load_images_and_labels(self, sample_size):
        images = []
        labels = []

        for root, _, files in os.walk(self.folder_path):
            for file_name in files:
                if file_name.endswith(('.jpg', '.png')):
                    file_path = os.path.join(root, file_name)
                    image = self.load_image(file_path)
                    images.append(image)
                    labels.append(os.path.basename(root))

        random_indices = random.sample(range(len(images)), sample_size)
        sampled_images = [images[i] for i in random_indices]
        sampled_labels = [labels[i] for i in random_indices]

        return sampled_images, sampled_labels
