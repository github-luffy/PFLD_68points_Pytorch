#-*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import shutil


def main(root_dir):
    files = ['300W/train_data/list.txt', 'WFLW/train_data/list.txt', '300VW/train_data/list.txt']
    dst_dir = os.path.join(root_dir, 'train_data')
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    dst_path = os.path.join(dst_dir, 'list.txt')
    
    with open(dst_path, 'w') as fw:
        for file_path in files:
            fp = os.path.join(root_dir, file_path)
            if os.path.isfile(fp):
                with open(fp, 'r') as fr:
                    lines = fr.readlines()
                    for index, line in enumerate(lines):
                        fw.write(line)
    
    test_files = ['300W/test_data/list.txt', 'WFLW/test_data/list.txt', '300VW/test_data/list.txt']
    test_dst_dir = os.path.join(root_dir, 'test_data')
    if not os.path.exists(test_dst_dir):
        os.mkdir(test_dst_dir)
    test_dst_path = os.path.join(test_dst_dir, 'list.txt')
    with open(test_dst_path, 'w') as fw:
        for file_path in test_files:
            fp = os.path.join(root_dir, file_path)
            if os.path.isfile(fp):
                with open(fp, 'r') as fr:
                    lines = fr.readlines()
                    for index, line in enumerate(lines):
                        fw.write(line)


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))

    print(root_dir)
    main(root_dir)
    # w_file = os.path.join(root_dir, 'data/300W/train_data/list.txt')
    # wflw_file = os.path.join(root_dir, 'data/WFLW/train_data/list.txt')
    # vw_file = os.path.join(root_dir, 'train_data/list.txt')
    # dst_file = 
    # path = os.path.join(root_dir, 'train_data/imgs')
    # print(w_file)
    # print(vw_file)
    # print(wflw_file)
    # print(dst_file)
    # all_lines = []
    # with open(w_file, 'r') as w_fr:
    #     lines = w_fr.readlines()
    #     for index, line in enumerate(lines):
    #         line_split = line.strip('\t\n').split(' ', 1)
    #         image_name = line_split[0].rsplit('/', 1)[-1]
    #         image_path = os.path.join(path, image_name)
    #         all_lines.append(image_path + ' ' + line_split[1])

    # with open(wflw_file, 'r') as wflw_fr:
    #     lines = wflw_fr.readlines()
    #     for index, line in enumerate(lines):
    #         line_split = line.strip('\t\n').split(' ', 1)
    #         image_name = line_split[0].rsplit('/', 1)[-1]
    #         image_path = os.path.join(path, image_name)
    #         all_lines.append(image_path + ' ' + line_split[1])
    # with open(vw_file, 'r') as vw_fr:
    #     lines = vw_fr.readlines()
    #     for index, line in enumerate(lines):
    #         line_split = line.strip('\t\n').split(' ', 1)
    #         image_name = line_split[0].rsplit('/', 1)[-1]
    #         image_path = os.path.join(path, image_name)
    #         all_lines.append(image_path + ' ' + line_split[1])
    # index_list = np.arange(len(all_lines))  # 生成下标
    # np.random.shuffle(index_list)
    # with open(dst_file, 'w') as fw:
    #     for i in index_list:
    #         fw.write(all_lines[i])
    #         fw.write('\n')
    # # with open(dst_file, 'w') as fw:
    # #     with open(w_file, 'r') as w_fr:
    # #         lines = w_fr.readlines()
    # #         for index, line in enumerate(lines):
    # #             line_split = line.strip('\t\n').split(' ', 1)
    # #             image_name = line_split[0].rsplit('/', 1)[-1]
    # #             image_path = os.path.join(path, image_name)
    # #             fw.write(image_path+' ')
    # #             fw.write(line_split[1])
    # #             fw.write('\n')
    # #     with open(wflw_file, 'r') as wflw_fr:
    # #         lines = wflw_fr.readlines()
    # #         for index, line in enumerate(lines):
    # #             line_split = line.strip('\t\n').split(' ', 1)
    # #             image_name = line_split[0].rsplit('/', 1)[-1]
    # #             image_path = os.path.join(path, image_name)
    # #             fw.write(image_path + ' ')
    # #             fw.write(line_split[1])
    # #             fw.write('\n')
    # #     with open(vw_file, 'r') as vw_fr:
    # #         lines = vw_fr.readlines()
    # #         for index, line in enumerate(lines):
    # #             line_split = line.strip('\t\n').split(' ', 1)
    # #             image_name = line_split[0].rsplit('/', 1)[-1]
    # #             image_path = os.path.join(path, image_name)
    # #             fw.write(image_path + ' ')
    # #             fw.write(line_split[1])
    # #             fw.write('\n')
    # print('end')