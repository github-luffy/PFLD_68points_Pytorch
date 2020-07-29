import cv2
import sys
import numpy as np
import datetime
import os
import glob
from .retinaface import RetinaFace


# thresh = 0.8
# scales = [1024, 1980]
#
# count = 1
#
# gpuid = 0
# #detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
# detector = RetinaFace('./mnet.25/mnet.25', 0, gpuid, 'net3')
#
# img = cv2.imread('test1.jpg')
# # print(img.shape)
# im_shape = img.shape
# target_size = scales[0]
# max_size = scales[1]
# im_size_min = np.min(im_shape[0:2])
# im_size_max = np.max(im_shape[0:2])
# #im_scale = 1.0
# #if im_size_min>target_size or im_size_max>max_size:
# im_scale = float(target_size) / float(im_size_min)
# # prevent bigger axis from being more than max_size:
# if np.round(im_scale * im_size_max) > max_size:
#     im_scale = float(max_size) / float(im_size_max)
#
# print('im_scale', im_scale)
#
# scales = [im_scale]
# flip = False
#
# for c in range(count):
#   faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
#   print(c, faces.shape, landmarks.shape)
#
# if faces is not None:
#   print('find', faces.shape[0], 'faces')
#   for i in range(faces.shape[0]):
#     #print('score', faces[i][4])
#     box = faces[i].astype(np.int)
#     #color = (255,0,0)
#     color = (0, 0, 255)
#     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
#     if landmarks is not None:
#       landmark5 = landmarks[i].astype(np.int)
#       #print(landmark.shape)
#       for l in range(landmark5.shape[0]):
#         color = (0, 0, 255)
#         if l == 0 or l == 3:
#           color = (0, 255, 0)
#         cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
#
#   filename = './detector_test2.jpg'
#   print('writing', filename)
#   cv2.imwrite(filename, img)


thresh = 0.6
scales = [120, 160]

model_path, _ = os.path.split(os.path.realpath(__file__))
model_path = os.path.join(model_path, 'mnet.25/mnet.25')
detector = RetinaFace(model_path, 0, ctx_id=0, network='net3')


def predict(image):
  im_shape = image.shape
  target_size = scales[0]
  max_size = scales[1]
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  im_scale = float(target_size) / float(im_size_min)
  if np.round(im_scale * im_size_max) > max_size:
    im_scale = float(max_size) / float(im_size_max)
  new_scales = [im_scale]

  faces, landmarks = detector.detect(image, thresh, scales=new_scales, do_flip=False)
  return faces, landmarks


# def read_camera():
#   thresh = 0.8
#   scales = [240, 320]
#   gpuid = 0
#
#   # detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
#   detector = RetinaFace('./mnet.25/mnet.25', 0, gpuid, 'net3')
#
#   cap = cv2.VideoCapture(0)
#   while True:
#     ret, image = cap.read()
#     if not ret:
#       break
#     im_shape = image.shape
#     target_size = scales[0]
#     max_size = scales[1]
#     im_size_min = np.min(im_shape[0:2])
#     im_size_max = np.max(im_shape[0:2])
#
#     im_scale = float(target_size) / float(im_size_min)
#     if np.round(im_scale * im_size_max) > max_size:
#       im_scale = float(max_size) / float(im_size_max)
#     new_scales = [im_scale]
#     flip = False
#     faces, landmarks = detector.detect(image, thresh, scales=new_scales, do_flip=flip)
#     if faces.shape[0] != 0:
#       print('find', faces.shape[0], 'faces')
#       for i in range(faces.shape[0]):
#         box = faces[i].astype(np.int)
#         color = (0, 0, 255)
#         cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
#         if landmarks is not None:
#           landmark5 = landmarks[i].astype(np.int)
#           for l in range(landmark5.shape[0]):
#             color = (0, 0, 255)
#             if l == 0 or l == 3:
#               color = (0, 255, 0)
#             cv2.circle(image, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
#     cv2.imshow('0', image)
#     if cv2.waitKey(10) == 27:
#       break
# read_camera()

if __name__ == '__main__':
    predict("./t1.jpg")



