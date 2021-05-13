from matplotlib import image
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import os
import numpy as np
import random


def data_transform(image_path):

    height = 128 # 180
    width = 128 # 180
    image_array = image.imread(image_path)

    drop_last_height = False
    drop_last_width = False

    height_slice = image_array.shape[0] - height
    if height_slice % 2 == 1:
        height_slice -= 1
        drop_last_height = True

    width_slice = image_array.shape[1] - width

    if height_slice <= 0 or width_slice <= 0:
        return None

    if width_slice % 2 == 1:
        width_slice -= 1
        drop_last_width = True

    cropped_array = image_array[int(height_slice / 2):-int(height_slice / 2),
                    int(width_slice / 2):-int(width_slice / 2), :]

    if drop_last_height:
        cropped_array = cropped_array[:-1, :, :]
    if drop_last_width:
        cropped_array = cropped_array[:, :-1, :]

    cropped_array = cropped_array / 255.
    cropped_array = cropped_array.reshape(3, 128, 128)
    cropped_array = torch.tensor(cropped_array, dtype=torch.float)

    return cropped_array


class ImagesDataset(Dataset):
    def __init__(self, img_dir):
        super(ImagesDataset, self).__init__()
        self.img_dir = img_dir
        self.classes = os.listdir(img_dir)
        self.num_classes = len(self.classes)
        self.class_number = {i : "images/"+class_ for i, class_ in zip(range(self.num_classes), self.classes)}
        self.keys_list = {}
        self.shuffle_keys()
        self.current_class = 0
        self.n_samples = min([len(self.keys_list[i]) for i in range(len(self.keys_list.keys()))])

    def __getitem__(self, index): # batch_size

        samples = []

        for i in range(self.num_classes):
            sample_index = self.keys_list[i][index]

            sample_path = os.path.join(self.class_number[i], self.classes[i].split("_")[0]+"_"+str(sample_index)+".jpg")
            sample = data_transform(sample_path)

            # if we grab an image that can't be cropped to 128x128, grab another image
            if sample is None:
                while sample is None:
                    sample_index = random.choice(range(1, len(self.keys_list[i])))
                    sample_path = os.path.join(self.class_number[i],
                                          self.classes[i].split("_")[0] + "_" + str(sample_index) + ".jpg")
                    sample = data_transform(sample_path)
            samples.append(sample)

        return samples

    def shuffle_keys(self):

        for i in range(self.num_classes):
            dir = self.class_number[i]
            dir_length = len(os.listdir(dir))

            self.keys_list[i] = list(range(1, dir_length+1))
            random.shuffle(self.keys_list[i])

    def __len__(self):
        return self.n_samples



