# VizWiz-VQA
This project participate the VizWiz VQA challenge. We try to use OCR information to improve the UNITER model.

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
