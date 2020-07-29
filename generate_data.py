import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset
import torchvision as tv

import cv2
import tensorflow as tf


class DataSet(Dataset):

    def __init__(self, file_list, image_channels, image_size, transforms=None,
                 loader=tv.datasets.folder.default_loader, is_train=True):
        self.file_list, self.landmarks, self.attributes = gen_data(file_list)
        self.image_channels = image_channels
        assert self.image_channels == 3
        self.image_size = image_size
        self.transforms = transforms
        self.loader = loader
        self.is_train = is_train

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        landmarks = self.landmarks[index]
        attributes = self.attributes[index]

        image = self.loader(file_name)

        # if self.is_train:
        #     assert image.size[0] == self.image_size

        if self.transforms is not None:
            image = self.transforms(image)

        # image = io.imread(file_name)
        # assert image.shape[2] == self.image_channels
        # transform.resize:dtype==float64,[0, 1]
        # image = transform.resize(image, (self.image_size, self.image_size))  # mode???
        # image = np.asarray(image, dtype=np.float32)

        return image, landmarks, attributes


# def DateSet(file_list, args):
#     file_list, landmarks, attributes = gen_data(file_list)
#
#     dataset = tf.data.Dataset.from_tensor_slices((file_list, landmarks, attributes))
#
#     def _parse_data(filename, landmarks, attributes):
#         file_contents = tf.read_file(filename)
#         image = tf.image.decode_png(file_contents, channels=args.image_channels)
#         image = tf.image.resize_images(image, (args.image_size, args.image_size), method=0)
#
#         # # 添加亮度,对比度的数据增强
#         # image = tf.image.random_brightness(image, max_delta=60)
#         # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
#
#         image = tf.cast(image, tf.float32)
#         image = image / 256.0
#         return (image, landmarks, attributes)
#
#     dataset = dataset.map(_parse_data)
#     dataset = dataset.shuffle(buffer_size=10000)
#     return dataset, len(file_list)


def gen_data(file_list):
    with open(file_list, 'r') as f:
        lines = f.readlines()

    filenames, landmarks,attributes = [], [], []
    for line in lines:
        line = line.strip().split()
        path = line[0]
        landmark = np.asarray(line[1:137], dtype=np.float32)
        attribute = np.asarray(line[137:], dtype=np.int32)
        filenames.append(path)
        landmarks.append(landmark)
        attributes.append(attribute)

    filenames = np.asarray(filenames, dtype=np.str)
    landmarks = np.asarray(landmarks, dtype=np.float32)
    attributes = np.asarray(attributes, dtype=np.int32)
    return (filenames, landmarks, attributes)


if __name__ == '__main__':
    file_list = 'data/train_data/list.txt'

    data_transforms = tv.transforms.Compose([
        tv.transforms.Resize((112, 112)),
        tv.transforms.ToTensor()
    ])

    train_dataset = DataSet(file_list, 3, 112, transforms=data_transforms)
    for i in range(len(train_dataset)):
        image, landmarks, attributes = train_dataset[i]
        # cv2.imshow('0', image)
        # cv2.waitKey(0)
        # image = np.asarray(image, dtype=np.float32)
        print(image.dtype)
        print(landmarks.dtype)
        print(attributes.dtype)
        exit(0)
    # file_list = 'data/train_data/list.txt'
    # filenames, landmarks, attributes = gen_data(file_list)
    # for i in range(len(filenames)):
    #     filename = filenames[i]
    #     landmark = landmarks[i]
    #     attribute = attributes[i]
    #     print(attribute)
    #     img = cv2.imread(filename)
    #     h, w, _ = img.shape
    #     landmark = landmark.reshape(-1, 2)*[h, w]
    #     for (x, y) in landmark.astype(np.int32):
    #         cv2.circle(img, (x, y), 1, (0, 0, 255))
    #     cv2.imshow('0', img)
    #     cv2.waitKey(0)
