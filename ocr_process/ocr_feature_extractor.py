from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import json
import torch
from PIL import Image
from glob import glob
import os

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")


def get_size(input_path):
    img_size = {}
    ips = sorted(glob(os.path.join(input_path, "*")))
    cnt = 0
    for ip in ips:
        img_name = ip.split("/")[-1]
        img = Image.open(ip)
        img_size[img_name] = img.size
    
    return img_size

def ocr_feature_extract(input_path, output_path, offset_path, img_size):
    feature_size = 768 + 8
    offset = {}
    with open(input_path, "r") as f:
        data = json.load(f)
    
    ocr_features = torch.zeros(len(data), 3, feature_size)
    
    off_idx = 0
    cnt = 0
    for img_id, boxes in tqdm(data.items()):
        offset[img_id] = off_idx
        W, H = img_size[img_id]
        ocr_feature = torch.zeros(3, feature_size)
        for idx, (box, sentence) in enumerate(boxes):
            a, _, b, _ = box
            x1, y1 = a
            x2, y2 = b
            w, h = b[0]-a[0], b[1]-a[1]
            x1, y1, x2, y2, w, h = x1/W, y1/H, x2/W, y2/H, w/W, h/H
            # print((idx+1)/3, x1, y1, x2, y2, w, h, w*h)
            if len(sentence) >= 512:
                sentence = sentence[:512]
            pos = torch.tensor([(idx+1)/3, x1, y1, x2, y2, w, h, w*h])
            encoded_input = tokenizer(sentence, return_tensors='pt')
            encoded_output = model(**encoded_input)
            bert_feature = encoded_output.last_hidden_state[0][0]
            
            # concat the feature
            feature = torch.cat((pos, bert_feature))
            ocr_feature[idx] = feature
        ocr_features[off_idx] = ocr_feature
        off_idx += 1
        cnt += 1
    
    torch.save(ocr_features, output_path)
    with open(offset_path, "w") as f:
        for img_id, off in offset.items():
            f.write("{}\t{}\n".format(img_id, off))

train_img_path = "/root/QianZe/VizWiz-VQA/train"
val_img_path = "/root/QianZe/VizWiz-VQA/val"

train_img_size = get_size(train_img_path)
val_img_size = get_size(val_img_path)

train_input_path = "/root/QianZe/Transformers-VQA/spelling_correction/train_newocr_oracle.json"
val_input_path = "/root/QianZe/Transformers-VQA/spelling_correction/val_newocr_oracle.json"
train_output_path = "/root/QianZe/Transformers-VQA/data/vizwiz/ocr_feat/train_ocr_oracle.pth"
val_output_path = "/root/QianZe/Transformers-VQA/data/vizwiz/ocr_feat/val_ocr_oracle.pth"
train_offset_path = "/root/QianZe/Transformers-VQA/data/vizwiz/ocr_feat/train_ocr_oracle_offset.txt"
val_offset_path = "/root/QianZe/Transformers-VQA/data/vizwiz/ocr_feat/val_ocr_oracle_offset.txt"

ocr_feature_extract(train_input_path, train_output_path, train_offset_path, train_img_size)
ocr_feature_extract(val_input_path, val_output_path, val_offset_path, val_img_size)