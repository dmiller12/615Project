from torch.utils.data import Dataset
import numpy as np


from PIL import Image


class MnistRotDataset(Dataset):
    def __init__(self, mode, transform=None):
        assert mode in ["train", "val", "test"]

        if mode == "train" or mode == "val":
            file = "data/raw/mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
        else:
            file = "data/raw/mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"

        self.transform = transform

        data = np.loadtxt(file, delimiter=" ")

        if mode == "train":
            self.labels = data[:10000, -1].astype(np.int64)
            self.num_samples = len(self.labels)
            self.images = data[:10000, :-1].reshape(-1, 28, 28).astype(np.float32)
        elif mode == "val":
            self.labels = data[10000:, -1].astype(np.int64)
            self.num_samples = len(self.labels)
            self.images = data[10000:, :-1].reshape(-1, 28, 28).astype(np.float32)
        else:
            self.labels = data[:, -1].astype(np.int64)
            self.num_samples = len(self.labels)
            self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
            

        self.images = np.pad(self.images, pad_width=((0, 0), (0, 1), (0, 1)), mode="edge")

        assert self.images.shape == (self.labels.shape[0], 29, 29)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)


class FilteredMNIST(Dataset):
    def __init__(self, mnist_dataset, digit):
        self.mnist_dataset = mnist_dataset
        self.filtered_indices = [i for i, (_, label) in enumerate(self.mnist_dataset) if label == digit]
    
    def __getitem__(self, index):
        return self.mnist_dataset[self.filtered_indices[index]]
    
    def __len__(self):
        return len(self.filtered_indices)


