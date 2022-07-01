from ast import dump
from paddleocr import PaddleOCR, draw_ocr
import os
from glob import glob
from tqdm import tqdm
import json
import cv2
from matplotlib import pyplot as plt
import numpy as np

def calc_single_image_ratio(boxes):
    if len(boxes) == 0:
        return -1
    
    ratio = 0
    for box in boxes:
        pos, text, _ = box
        x1, x2, x3, x4 = pos
        wide = x2[0] - x1[0]
        height = x3[1] - x2[1]
        if wide == 0.0:
            wide = 1
        if height == 0.0:
            height = 1
        tmp_ratio = wide / height
#         print(tmp_ratio)
        ratio += tmp_ratio
    
    average_ratio = ratio / len(boxes)
    
    return average_ratio

def rotate_detect_single_image(img_path, ocr):
    #     print(img_path)
    img_cv = cv2.imread(img_path)
    img_np = np.array(img_cv)
    
    drt_dict = {}
    sum_conf, average_angle = 0, 0
    for i in range(4):
        img_np_i = np.rot90(img_np, i)
        boxes = ocr.ocr(img_np_i, cls=True)
        ratio = calc_single_image_ratio(boxes)
        sum_angle, sum_conf, average_angle = 0, 0, 0
        if len(boxes) != 0:
            for box in boxes:
                pos, text, angle = box
                sum_conf += text[1]
                sum_angle += int(angle[0])
            average_angle = sum_angle / len(boxes)
        drt_dict[i] = [sum_conf, ratio, average_angle]
    
    true_drt = -1
    max_conf = 0
    for drt, drt_info in drt_dict.items():
        sum_conf, ratio, average_angle = drt_info
        if ratio <= 0.5:
            continue
        if average_angle > 90:
            continue
        if sum_conf > max_conf:
            true_drt = drt
            max_conf = sum_conf
    
    img_np_drt_true = np.rot90(img_np, true_drt)
    true_drt_info = ocr.ocr(img_np_drt_true, cls=True)

    H, W, C = img_np_drt_true.shape
    
    return true_drt, true_drt_info, (H, W)

if __name__ == "__main__":
    img_path = "/root/QianZe/VizWiz-VQA-tiny/test_rectify"
    ips = sorted(glob(os.path.join(img_path, "*")))
    # print(ips[:10])
    pred = {}

    for ip in tqdm(ips):
        img_name = ip.split("/")[-1].split(".")[0]
        max_drt = rotate_detect_single_image(ip)
        pred[img_name] = max_drt
    
    rectify_label_path = "/root/QianZe/VizWiz-VQA-tiny/test_rectify_label.txt"

    gt = {}
    with open(rectify_label_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if len(line) <= 1: continue
            image_id, label = line.strip().split("\t")
            gt[image_id] = int(label)

    acc = 0
    for image_id in gt.keys():
        if pred[image_id] == gt[image_id]:
            acc += 1
            
    acc = acc / len(gt)
    print(acc)