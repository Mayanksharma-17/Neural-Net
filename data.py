from pathlib import Path
from PIL import Image
import numpy as np
import struct
from array import array

class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)        

def save_images(images, labels, output_dir):
    output_dir = Path(output_dir)

    for i, (image, label) in enumerate(zip(images, labels)):
        folder = output_dir / str(label)
        folder.mkdir(parents=True, exist_ok=True)

        Image.fromarray(np.array(image, dtype=np.uint8)).save(
            folder / f"{i}.png"
        )

loader = MnistDataloader(
    "data/train-images.idx3-ubyte",
    "data/train-labels.idx1-ubyte",
    "data/t10k-images.idx3-ubyte",
    "data/t10k-labels.idx1-ubyte",
)

(x_train, y_train), (x_test, y_test) = loader.load_data()

save_images(x_train, y_train, "images/train")
save_images(x_test, y_test, "images/test")
