import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T

class Dataset(data.Dataset):
    def __init__(self, root, imgs, phase = "train"):
        self.phase = phase
        self.input_shape = (3, 128, 128)

        imgs = [os.path.join(root, img) for img in imgs]
        self.imgs = np.random.permutation(imgs)

        if self.phase == "train":
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean = [0.5], std = [0.5])
            ])
        else:
            self.transforms = T.Compose([
                T.ToTensor(),
                T.Normalize(mean = [0.5], std = [0.5])
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        img_path = splits[0]
        data = Image.open(img_path)
        data = self.transforms(data)
        label = np.int32(splits[1])
        return data.float(), label

    def __len__(self):
        return len(self.imgs)
