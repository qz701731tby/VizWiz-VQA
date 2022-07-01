from rectify_boxes import *
from box_connecter import boxes_connect, rect_rot90
import json

slope_thres = 0.15
char_thres = 0.4
overlap_thres = 0.5
y_dis_thres = 1.5

img_path = "/root/QianZe/VizWiz-VQA-tiny/test_rectify"
ips = sorted(glob(os.path.join(img_path, "*")))
# print(ips[:10])
ocr_boxes = {}
ocr_texts = {}
img_size = {}
drt_record = {}

for ip in tqdm(ips):
    img_id = ip.split("/")[-1].split(".")[0]
    true_drt, true_drt_info, img_shape = rotate_detect_single_image(ip)
    boxes, texts = [], []
    for box_info in true_drt_info:
        pos, text, _ = box_info
        boxes.append(pos)
        texts.append(text[0])
        
    ocr_boxes[img_id] = boxes
    ocr_texts[img_id] = texts
    drt_record[img_id] = true_drt
    img_size[img_id] = img_shape

new_ocr_results = {}
for img_id in tqdm(ocr_boxes.keys()):
    # connect neighbour boxes
    new_rects, new_texts = boxes_connect(ocr_boxes[img_id], ocr_texts[img_id], slope_thres, char_thres, overlap_thres, y_dis_thres)
    # rotate the boxes to original direction
    new_rects = rect_rot90(new_rects, drt_record[img_id], img_size[img_id])
    new_ocr_results[img_id] = [[rect, content] for rect, content in zip(new_rects, new_texts)]


output_path = "/root/QianZe/Transformers-VQA/data/vizwiz/ocr_result/test_ocr_output.json"
with open(output_path, "w") as f:
    json.dump(new_ocr_results, f, indent=4)