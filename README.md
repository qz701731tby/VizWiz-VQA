# VizWiz-VQA
This project participate the VizWiz VQA challenge. We try to use OCR information to improve the UNITER model.

[![OSCS Status](https://www.oscs1024.com/platform/badge/qz701731tby/VizWiz-VQA.svg?size=small)](https://www.oscs1024.com/project/qz701731tby/VizWiz-VQA?ref=badge_small)


# File Structure

```
project
│ README.md
│ vqa_model.py  
│ vqa_data.py
│ vqa_vizwiz.py
│ vqa.py
│
└───preprocess
│   │ data_process.ipynb
│   │ OCR_utils.py
│   │ stop_list_0.py
│
└───ocr_process
│   │ ocr_process.py
│   │ ocr_feature_extractor.py
│   │ box_connecter.py
│   │ rectify_boxes.py
│
└───src
│   │ entry.py
│   │ modeling.py
│   │ optimization.py
│   │ tokenization.py
│   │ file_utils.py
│
└───models
│   └───paddleOCR_20220802
│   └───pretrained
│   │    │ uniter-base.pt
│
└───data
│   └───vizwiz_imgfeat
│   └───vqa_label
│   │    │ train.json
│   │    │ val.json
│   │    │ trainval_ans2label.json
│   │    │ trainval_label2ans.json
│   │
│   └───paddle_ocr_feat
```

# VQA data 
https://vizwiz.org/tasks-and-datasets/vqa/

# training
### 1. image feature extract
For extract methods, please refer to https://github.com/airsplay/py-bottom-up-attention.
### 2. OCR (under `./ocr_process`)
In this part, we do OCR and box merge, `img_path` is the image folder you need to process:

` python ocr_process.py --img_path ./VizWiz/train --model en`

### 3. VQA label process (under `./preprocess`)
This part contains label selection (soft label and hard label) and OCR boxes selection. For details, please refer to `data_process.ipynb`

### 4. OCR feature extract (under `./ocr_process`)
We extract the feature for selected boxes in part 3 with BERT model. The OCR feature contains position info `[i, x1, y1, x2, y2, w, h, w*h]` and OCR sentence BERT [CLS] feature.

`python ocr_feature_extractor.py`

### 5. train
If you change the data path, please change the corresponding code in `vqa_vizwiz.py`:
```
VQA_DATA_ROOT = 'data/vizwiz/use_paddle_ocr_en_0704/'
VIZWIZ_IMGFEAT_ROOT = '/data_zt/VQA/vizwiz_imgfeat'
VIZWIZ_OCRFEAT_ROOT = 'data/vizwiz/paddle_ocr_feat/en_oracle/'
```
Then run the following command line:

`python vqa.py --model uniter --epochs 15 --max_seq_length 20 --load_pretrained models/pretrained/uniter-base.pt --output models/paddleOCR_20220802/`


# performance
with ocr feature (5% better than non-ocr)
| accuracy      | yes | other | number | unanswerable | ocr | average|
| ----------- | ----------- | ---- | ----| ----| ---- | --- |
| train   |  73.85 | 64.87 | 74.80 | 82.12 | 46.34 | 70.20 |
| val   | 53.09 | 40.88 | 36.46 | 79.28 | 32.99 | 54.08 |

# Acknowledgment
- label selection: https://github.com/DenisDsh/VizWiz-VQA-PyTorch
- image feature extraction: https://github.com/airsplay/py-bottom-up-attention
- ocr: https://github.com/PaddlePaddle/PaddleOCR
- UNITER base model: https://github.com/YIKUAN8/Transformers-VQA
