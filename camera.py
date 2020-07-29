from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('pid: {}     GPU: {}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))
import tensorflow as tf
import numpy as np
import cv2
import time
import dlib
from RetinaFaceMaster.test import predict
from mtcnn.detect_face import MTCNN
from model2 import MobileNetV2, BlazeLandMark
import torch


def read_images():
    meta_file = './test_models2/model.meta'
    ckpt_file = './test_models2/model.ckpt-187'
    main_dir = './test_models2/DSM_lina/Lina15'
    landmark_file = '%s/landmarks1.txt' % main_dir
    result_image_path = '%s/result_image' % main_dir
    if not os.path.exists(result_image_path):
        os.mkdir(result_image_path)
    image_size = 112

    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            print('Loading feature extraction model.')
            saver = tf.train.import_meta_graph(meta_file)
            saver.restore(tf.get_default_session(), ckpt_file)

            graph = tf.get_default_graph()
            images_placeholder = graph.get_tensor_by_name('image_batch:0')
            phase_train_placeholder = graph.get_tensor_by_name('phase_train:0')
            landmark_total = graph.get_tensor_by_name('pfld_inference/fc/BiasAdd:0')
            mtcnn = MTCNN()
            sum_time = 0.
            num_face_box = 0
            with open(landmark_file, 'r') as fr:
                lines = fr.readlines()
                for index, line in enumerate(lines):
                    line_split = line.strip('\t\n').split(' ', 1)
                    image_path = '%s/%s' % (main_dir, line_split[0])
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    height, width, _ = image.shape
                    boxes = mtcnn.predict(image)
                    # boxes, _ = predict(image)
                    for box in boxes:
                        start_time = time.time()
                        x1, y1, x2, y2 = (box[:4] + 0.5).astype(np.int32)
                        w = x2 - x1 + 1
                        h = y2 - y1 + 1

                        size = int(max([w, h]) * 1.1)
                        cx = x1 + w // 2
                        cy = y1 + h // 2
                        x1 = cx - size // 2
                        x2 = x1 + size
                        y1 = cy - size // 2
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
                        if dx > 0 or dy > 0 or edx > 0 or edy > 0:
                            cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
                        cropped = cv2.resize(cropped, (image_size, image_size))

                        input = cv2.resize(cropped, (image_size, image_size))
                        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
                        input = input.astype(np.float32) / 256.0
                        input = np.expand_dims(input, 0)

                        feed_dict = {
                            images_placeholder: input,
                            phase_train_placeholder: False
                        }

                        pre_landmarks = sess.run(landmark_total, feed_dict=feed_dict)
                        pre_landmark = pre_landmarks[0]
                        pre_landmark = pre_landmark.reshape(-1, 2) * [size, size]

                        sum_time += (time.time() - start_time)
                        num_face_box += 1

                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
                        for (x, y) in pre_landmark.astype(np.int32):
                            cv2.circle(image, (x1 + x, y1 + y), 1, (0, 0, 255), 2)

                    # points = line_split[1].split(',')
                    #
                    # for i in range(68):
                    #     cv2.circle(image, (int(float(points[i*2])), int(float(points[i*2+1]))), 1, (255, 0, 0), 2)

                    # for i in range(68):
                    #     cv2.circle(image, (int(float(points[i*2]) * width), int(float(points[i*2+1])*height)),
                    #                1, (255, 0, 0), 2)
                    # cv2.imwrite('%s/%s' % (result_image_path, line_split[0].rsplit('/', 1)[1]), image)
                    cv2.imshow('0', image)
                    if cv2.waitKey(10) == 27:
                        break
    # img_path = './test_models2/DSM_lina/01.bmp'
    #
    # image_size = 112
    # with tf.Graph().as_default():
    #     with tf.Session() as sess:
    #         print('Loading feature extraction model.')
    #         saver = tf.train.import_meta_graph(meta_file)
    #         saver.restore(tf.get_default_session(), ckpt_file)
    #
    #         graph = tf.get_default_graph()
    #         images_placeholder = graph.get_tensor_by_name('image_batch:0')
    #         phase_train_placeholder = graph.get_tensor_by_name('phase_train:0')
    #         landmark_total = graph.get_tensor_by_name('pfld_inference/fc/BiasAdd:0')
    #
    #
    #         image = cv2.imread(img_path)
    #         if image is None:
    #             print('Error:image is None!')
    #         height, width, _ = image.shape
    #         boxes, _ = predict(image)
    #         for box in boxes:
    #             x1, y1, x2, y2 = (box[:4] + 0.5).astype(np.int32)
    #             w = x2 - x1 + 1
    #             h = y2 - y1 + 1
    #
    #             size = int(max([w, h]) * 1.1)
    #             cx = x1 + w // 2
    #             cy = y1 + h // 2
    #             x1 = cx - size // 2
    #             x2 = x1 + size
    #             y1 = cy - size // 2
    #             y2 = y1 + size
    #
    #             dx = max(0, -x1)
    #             dy = max(0, -y1)
    #             x1 = max(0, x1)
    #             y1 = max(0, y1)
    #
    #             edx = max(0, x2 - width)
    #             edy = max(0, y2 - height)
    #             x2 = min(width, x2)
    #             y2 = min(height, y2)
    #
    #             cropped = image[y1:y2, x1:x2]
    #             if dx > 0 or dy > 0 or edx > 0 or edy > 0:
    #                 cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
    #             cropped = cv2.resize(cropped, (image_size, image_size))
    #
    #             input = cv2.resize(cropped, (image_size, image_size))
    #             input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    #             input = input.astype(np.float32) / 256.0
    #             input = np.expand_dims(input, 0)
    #
    #             feed_dict = {
    #                 images_placeholder: input,
    #                 phase_train_placeholder: False
    #             }
    #             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
    #             pre_landmarks = sess.run(landmark_total, feed_dict=feed_dict)
    #             pre_landmark = pre_landmarks[0]
    #
    #             pre_landmark = pre_landmark.reshape(-1, 2) * [size, size]
    #             for (x, y) in pre_landmark.astype(np.int32):
    #                 cv2.circle(image, (x1 + x, y1 + y), 1, (0, 0, 255), 2)
    #
    #         cv2.imshow('0', image)
    #         cv2.waitKey(0)
            average_time = sum_time*1000.0/num_face_box
            print("nums:{}, sum time:{:.3f}s, avg time:{:.3f}ms\n".format(num_face_box, sum_time, average_time))


def main():
    ckpt_file = './test_models2/model_37.pth'
    videl_file = './test_models2/DSM_lina/Lina22/lina22.mp4'
    image_size = 112  # 112

    # coefficient = 0.25
    # print(coefficient)
    # num_of_channels = [int(64 * coefficient), int(128 * coefficient), int(16 * coefficient), int(32 * coefficient),
    #                    int(128 * coefficient)]
    # model = MobileNetV2(num_of_channels=num_of_channels, nums_class=136)
    model = BlazeLandMark(nums_class=136)
    model = torch.load(ckpt_file)
    model.eval()

    cap = cv2.VideoCapture(0)
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

            cropped = image[y1:y2, x1:x2]
            if dx > 0 or dy > 0 or edx > 0 or edy > 0:
                cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
            cropped = cv2.resize(cropped, (image_size, image_size))

            input = cv2.resize(cropped, (image_size, image_size))
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            input = input.astype(np.float32) / 256.0
            input = np.expand_dims(input, 0)
            input = torch.Tensor(input.transpose((0, 3, 1, 2)))

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
            pre_landmarks, _ = model(input.cuda())
            pre_landmark = pre_landmarks[0].cpu().detach().numpy()

            pre_landmark = pre_landmark.reshape(-1, 2) * [size, size]
            for (x, y) in pre_landmark.astype(np.int32):
                cv2.circle(image, (x1 + x, y1 + y), 1, (0, 0, 255), 2)
        # image = cv2.resize(image, (width, height))
        cv2.imshow('0', image)
        if cv2.waitKey(10) == 27:
            break

if __name__ == '__main__':
    main()
    # read_images()
