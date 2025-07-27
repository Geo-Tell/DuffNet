import cv2
import os
# 打开文件并按行读取内容
file_path = '/home/ch/Data/NWPU_patch512/val_label.txt'
out_path1 = '/home/ch/Data/NWPU_patch512/val_imgs'
with open(file_path, 'r', encoding='utf-8') as file:
    # 读取所有行，并返回一个包含各行内容的列表
    lines = file.readlines()
    separator = ' '
    for i in range(len(lines)):
        lines[i] = lines[i].split(separator, 1)[0]
    #print(lines[0:10])
    #lines = [i[:-1] for i in lines]
    #print(lines[0])
for i in range(len(lines)):
    image = cv2.imread(lines[i])
    # print(i)
    # print(lines[i])
    # print(image)
    dst = os.path.join(os.path.abspath(out_path1), '' + str(i) + '.png')
    #print(dst)
    cv2.imwrite(dst, image)
    