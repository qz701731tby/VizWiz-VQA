from ast import dump
from paddleocr import PaddleOCR, draw_ocr
import os
from glob import glob
from tqdm import tqdm
import json

# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='ch') # need to run only once to download and load model into memory

# img_path = "../VizWiz-VQA/train"
img_path = "../VizWiz-VQA/val"
ips = sorted(glob(os.path.join(img_path, "*")))
# print(ips[:10])
results = {}

# test_img_path = "../VizWiz-VQA/val/VizWiz_val_00000031.jpg"
# result = ocr.ocr(test_img_path, cls=True)
# print(result)

cnt = 0
for ip in tqdm(ips):
    img_name = ip.split("/")[-1].split(".")[0]
    result = ocr.ocr(ip, cls=True)
    results[img_name] = result
    # if cnt <= 10:
    #     for line in result:
    #         print(line)
    cnt += 1

# output_path = "../VizWiz-VQA-tiny/val_tiny_ocr_result.json"
output_path = "./data/vizwiz/ocr_result/val_ocr_result_ch_PP-OCRv3.json"

import numpy
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
            numpy.uint16,numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32, 
            numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)): # add this line
            return obj.tolist() # add this line
        return json.JSONEncoder.default(self, obj) 

with open(output_path, "w") as f:
    json.dump(results, f, indent=4, cls=NpEncoder)
