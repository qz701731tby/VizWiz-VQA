# VizWiz-VQA
This project participate the VizWiz VQA challenge. We try to use OCR information to improve the UNITER model.



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



# data 
https://vizwiz.org/tasks-and-datasets/vqa/

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
