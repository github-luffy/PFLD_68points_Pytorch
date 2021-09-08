from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('pid: {}     GPU: {}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))
import numpy as np
import cv2
import time
from RetinaFaceMaster.test import predict
from mtcnn.detect_face import MTCNN
from model2 import MobileNetV2, BlazeLandMark
import torch


def test_images():
    model_path = './models2/model0/model_499.pth'
    images_dir = './test_images_2/'  #'./data/test_data/imgs/'
    image_size = 112  # 112

    model = BlazeLandMark(nums_class=136)
    model = torch.load(model_path)
    model.eval()
    image_files = os.listdir(images_dir)
    for index, image_file in enumerate(image_files):
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        if image is None:
            print(image_path)
            continue
        # boxes = mtcnn.predict(image)
        # image = cv2.resize(image, (width//2, height//2))
        boxes, _ = predict(image)
        for box in boxes:
            x1, y1, x2, y2 = (box[:4]+0.5).astype(np.int32)
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            size = int(max([w, h])*1.1)
            cx = x1 + w//2
            cy = y1 + h//2
            x1 = cx - size//2
            x2 = x1 + size
            y1 = cy - size//2
            y2 = y1 + size

            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)
            cropped = image[y1:y2, x1:x2]
            box_w = x2-x1
            box_h = y2-y1
            #print(dy, edy, dx, edx)
            if dx > 0 or dy > 0 or edx > 0 or edy > 0:
                cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)  # top,bottom,left,right
            # cropped = cv2.resize(cropped, (image_size, image_size))

            input = cv2.resize(cropped, (image_size, image_size))
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            input = input.astype(np.float32) / 255.0
            input = np.expand_dims(input, 0)
            input = torch.Tensor(input.transpose((0, 3, 1, 2)))

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
            pre_landmarks, _ = model(input.cuda())
            pre_landmark = pre_landmarks[0].cpu().detach().numpy()
            pre_landmark = pre_landmark.reshape(-1, 2)
            pre_landmark = pre_landmark*[box_w+dx+edx, box_h+dy+edy]-[dx, dy]
            for (x, y) in pre_landmark.astype(np.int32):
                cv2.circle(image, (x1 + x, y1 + y), 2, (0, 0, 255), 2)
        # image = cv2.resize(image, (width, height))
        cv2.imshow('0', image)
        if cv2.waitKey(0) == 27:
            break


def main():
    model_file = './models2/model0/model_499.pth'
    videl_file = './dfl.mov'
    image_size = 112  # 112

    # coefficient = 0.25
    # print(coefficient)
    # num_of_channels = [int(64 * coefficient), int(128 * coefficient), int(16 * coefficient), int(32 * coefficient),
    #                    int(128 * coefficient)]
    # model = MobileNetV2(num_of_channels=num_of_channels, nums_class=136)
    model = BlazeLandMark(nums_class=136)
    model = torch.load(model_file)
    model.eval()

    cap = cv2.VideoCapture(videl_file)
    while True:
        ret, image = cap.read()
        height, width, _ = image.shape
        if not ret:
            break
        # boxes = mtcnn.predict(image)
        # image = cv2.resize(image, (width//2, height//2))
        boxes, _ = predict(image)
        for box in boxes:
            x1, y1, x2, y2 = (box[:4]+0.5).astype(np.int32)
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            size = int(max([w, h])*1.1)
            cx = x1 + w//2
            cy = y1 + h//2
            x1 = cx - size//2
            x2 = x1 + size
            y1 = cy - size//2
            y2 = y1 + size

            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)
            box_w = x2-x1
            box_h = y2-y1
            cropped = image[y1:y2, x1:x2]
            if dx > 0 or dy > 0 or edx > 0 or edy > 0:
                cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
            # cropped = cv2.resize(cropped, (image_size, image_size))

            input = cv2.resize(cropped, (image_size, image_size))
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            input = input.astype(np.float32) / 255.0
            input = np.expand_dims(input, 0)
            input = torch.Tensor(input.transpose((0, 3, 1, 2)))

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
            pre_landmarks, _ = model(input.cuda())
            pre_landmark = pre_landmarks[0].cpu().detach().numpy()

            pre_landmark = pre_landmark.reshape(-1, 2)
            pre_landmark = pre_landmark*[box_w+dx+edx, box_h+dy+edy]-[dx, dy]
            for (x, y) in pre_landmark.astype(np.int32):
                cv2.circle(image, (x1 + x, y1 + y), 2, (0, 0, 255), 2)
        # image = cv2.resize(image, (width, height))
        cv2.imshow('0', image)
        if cv2.waitKey(10) == 27:
            break

if __name__ == '__main__':
    #main()
    test_images()
