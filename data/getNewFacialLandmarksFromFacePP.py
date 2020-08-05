import cv2
import requests
import json
import os
import base64
import numpy as np

def getFacailLandmarksFromFacepp():
    url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
    image_dir = '/data/new_face_images'  # Dir for New Face Images   
    images = os.listdir(image_dir)

    for index, image_name in enumerate(images):
        image_path = os.path.join(image_dir, image_name)
        #print(image_path)
        img_file = {'image_file':open(image_path, 'rb')}
        payload = {'api_key':'your_api_key',
                    'api_secret':'your_api_secret',
                    'return_landmark':2}
        r = requests.post(url, files=img_file, data=payload)
        if r:    
            data = json.loads(r.text)
            image = cv2.imread(image_path)
            h,w,_ = image.shape
            if 'faces' not in data.keys() or len(data['faces']) == 0:
                print(image_path)
            else:
                assert len(data['faces']) == 1
                for i in range(len(data['faces'])):
                    face = data['faces'][i]
                    width = face['face_rectangle']['width']
                    top = face['face_rectangle']['top']
                    height = face['face_rectangle']['height']
                    left = face['face_rectangle']['left']
                    cv2.rectangle(image, (left, top),(left+width, top+height), (0, 255, 0), 1)
                    print(face['landmark'])
                    for j in face['landmark']:
                        point = face['landmark'][j]
                        x = point['x']
                        y = point['y']
                        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
                #cv2.imwrite(os.path.join(new_image_save_dir, image_name), image)
                cv2.imshow("image", image)
                cv2.waitKey(0)        
        if (index+1) % 100 == 0 or (index+1) == len(images):
            print('%d done', index+1)           


if __name__ == '__main__':
    main()


