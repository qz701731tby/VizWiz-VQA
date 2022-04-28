from ast import dump
import easyocr
import os
from glob import glob
from tqdm import tqdm
import json

reader = easyocr.Reader(['en'])
img_path = "../VizWiz-VQA/train"
ips = sorted(glob(os.path.join(img_path, "*")))
print(ips[:10])
results = {}

cnt = 0
for ip in tqdm(ips):
    # if cnt > 10: break
    img_name = ip.split("/")[-1]
    result = reader.readtext(ip, paragraph=True)
    results[img_name] = result
    if cnt <= 10:
        print(result)
    cnt += 1

output_path = "./data/vizwiz/train_ocr_result.json"

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
