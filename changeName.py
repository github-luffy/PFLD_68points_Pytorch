#-*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import shutil


def main():
    # root_dir = os.path.dirname(os.path.realpath(__file__))
    # print(root_dir)
    # image_file = '%s/data/test_data/list.txt' % root_dir
    # dst_image_file = '%s/data/train_data/list.txt' % root_dir
    # with open(dst_image_file, 'a+') as fw:
    #     with open(image_file, 'r') as fr:
    #         lines = fr.readlines()
    #         for index, line in enumerate(lines):
    #             line_data = line.strip('\t\n').split(' ', 1)
    #             image_name = line_data[0].split('/')[-1]
    #             split_name = image_name.split('_')
    #             dst_image_name = '%s_%08d_%s' % (95192 + int(split_name[0]), 95192 + int(split_name[1]), split_name[2])
    #             print(dst_image_name)
    #             fw.write('%s/%s ' % ('/home/dsai/datadisk/pfld68/PFLD-master/data/train_data/imgs', dst_image_name))
    #             fw.write(line_data[1])
    #             fw.write('\n')
    pass


if __name__ == '__main__':

    root_dir = os.path.dirname(os.path.realpath(__file__))
    print(root_dir)
    images_dir = '%s/data/test_data/imgs' % root_dir
    images = os.listdir(images_dir)
    images.sort(key=lambda x: int(x.split('_')[1]))
    print(len(images))
    for index, image_name in enumerate(images):
        split_name = image_name.split('_')
        dst_image_name = '%s_%08d_%s' % (95192 + int(split_name[0]), 95192 + int(split_name[1]), split_name[2])
        # print(image_name, ',', dst_image_name)
        image = cv2.imread('%s/%s' % (images_dir, image_name))
        cv2.imwrite('%s/data/train_data/imgs/%s' % (root_dir, dst_image_name), image)
        if (index + 1) % 1000 == 0:
            print('Done for %d' % (index + 1))
    print('end')
