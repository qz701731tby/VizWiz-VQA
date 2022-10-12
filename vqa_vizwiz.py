# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle
import base64
from tokenize import String

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.
VQA_DATA_ROOT = 'data/vizwiz/use_paddle_ocr_en_0704/'
VIZWIZ_IMGFEAT_ROOT = '/data_zt/VQA/vizwiz_imgfeat'
VIZWIZ_OCRFEAT_ROOT = 'data/vizwiz/paddle_ocr_feat/en_oracle/'
SPLIT2NAME = {
    'train': 'train',
    'val': 'val',
    'test': 'test',
}

class VizWizVQADataset:
    """
    A VizWiz data example in json file:
        {
            "answer_type": "unanswerable",
            "img_id": "VizWiz_val_00000000",
            "label": {
                "unanswerable": 1
            },
            "question_id": 0,
            "sent": "Ok. There is another picture I hope it is a better one."
        }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("{}{}.json".format(VQA_DATA_ROOT, SPLIT2NAME[split]))))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open("{}trainval_ans2label.json".format(VQA_DATA_ROOT)))
        self.label2ans = json.load(open("{}trainval_label2ans.json".format(VQA_DATA_ROOT)))
        print("len of label: ", len(self.label2ans))
        assert len(self.ans2label) == len(self.label2ans)

        self.train_ocr_path = os.path.join(VIZWIZ_OCRFEAT_ROOT, '%s_ocr.pth' % ('train'))
        self.val_ocr_path = os.path.join(VIZWIZ_OCRFEAT_ROOT, '%s_ocr.pth' % ('val'))

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class VizWizVQATorchDataset(Dataset):
    def __init__(self, dataset: VizWizVQADataset, model = 'uniter'):
        super().__init__()
        self.raw_dataset = dataset
        self.train_ocr_data = self._loadOcrFeat(dataset.train_ocr_path)
        self.val_ocr_data = self._loadOcrFeat(dataset.val_ocr_path)
        self.model = model
        if args.tiny:
            topk = TINY_IMG_NUM
            self.raw_dataset.data = self.raw_dataset.data[:topk]
        elif args.fast:
            topk = FAST_IMG_NUM
            self.raw_dataset.data = self.raw_dataset.data[:topk]
        else:
            topk = None

        self.offset = {}
        for split in self.raw_dataset.splits:
            f = open(os.path.join(VIZWIZ_IMGFEAT_ROOT, '%s_offset.txt' % (SPLIT2NAME[split])))
            offset = f.readlines()
            for l in offset:
                self.offset[l.split('\t')[0]] = int(l.split('\t')[1].strip())
        
        self.ocr_offset = {}
        for split in self.raw_dataset.splits:
            f = open(os.path.join(VIZWIZ_OCRFEAT_ROOT, '%s_ocr_offset.txt' % (SPLIT2NAME[split])))
            offset = f.readlines()
            for l in offset:
                self.ocr_offset[l.split('\t')[0]] = int(l.split('\t')[1].strip())
        
        f = open(os.path.join(VIZWIZ_IMGFEAT_ROOT, '%s_d2obj36_batch.tsv' % (SPLIT2NAME['train'])))
        self.train_lines = f.readlines()
        f = open(os.path.join(VIZWIZ_IMGFEAT_ROOT, '%s_d2obj36_batch.tsv' % (SPLIT2NAME['val'])))
        self.val_lines = f.readlines()
        # f = open(os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_d2obj36_batch.tsv' % (SPLIT2NAME['test'])))
        # self.val_lines = f.readlines()

        self.data = self.raw_dataset.data

        print("Use %d data in torch dataset" % (len(self.data)))
        print()


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item: int):
        datum = self.data[item]
        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']
        answer_type = datum['answer_type']
        ocr_feats = None 
        ocr_boxes = None 

        img_offset = self.offset[img_id]
        img_split = img_id[7:9]
        # print("img_split: ", img_split)
        if(img_split == 'tr'):
            img_info = self.train_lines[img_offset]
            ocr_boxes, ocr_feats = self._decodeOcrFeat(self.ocr_offset[img_id], mode="train")
        elif(img_split == 'va'):
            img_info = self.val_lines[img_offset]
            ocr_boxes, ocr_feats = self._decodeOcrFeat(self.ocr_offset[img_id], mode="val")

        assert img_info.startswith('VizWiz') and img_info.endswith('\n'), 'Offset is inappropriate'
        img_info = img_info.split('\t')

        decode_img = self._decodeIMG(img_info)
        img_h = decode_img[0]
        img_w = decode_img[1]
        feats = decode_img[-1].copy()
        boxes = decode_img[-2].copy()
        del decode_img


        # Normalize the boxes (to 0 ~ 1)
        if self.model == 'uniter':
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            boxes = self._uniterBoxes(boxes)
            np.testing.assert_array_less(boxes, 1+1e-5)
            np.testing.assert_array_less(-boxes, 0+1e-5)
        else:
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h

            np.testing.assert_array_less(boxes, 1+1e-5)
            np.testing.assert_array_less(-boxes, 0+1e-5)


        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ocr_feats, ocr_boxes, ques, target, answer_type, img_id
            # return ques_id, feats, boxes, ques, target, answer_type, img_id
        else:
            return ques_id, feats, boxes, ocr_feats, ocr_boxes, ques, answer_type, img_id
            # return ques_id, feats, boxes, ques, answer_type, img_id

    def _decodeIMG(self, img_info):
        img_h = int(img_info[1])
        img_w = int(img_info[2])
        boxes = img_info[-2]
        boxes = np.frombuffer(base64.b64decode(boxes), dtype=np.float32)
        boxes = boxes.reshape(36,4)
        boxes.setflags(write=False)
        feats = img_info[-1]
        feats = np.frombuffer(base64.b64decode(feats), dtype=np.float32)
        feats = feats.reshape(36,-1)
        feats.setflags(write=False)
        return [img_h, img_w, boxes, feats]
    
    def _uniterBoxes(self, boxes):
        new_boxes = np.zeros((boxes.shape[0],7),dtype='float32')
        # new_boxes = np.zeros((boxes.shape[0],7),dtype='float32')
        new_boxes[:,1] = boxes[:,0]
        new_boxes[:,0] = boxes[:,1]
        new_boxes[:,3] = boxes[:,2]
        new_boxes[:,2] = boxes[:,3]
        new_boxes[:,4] = new_boxes[:,3]-new_boxes[:,1]
        new_boxes[:,5] = new_boxes[:,2]-new_boxes[:,0]
        new_boxes[:,6] = new_boxes[:,4]*new_boxes[:,5]
        return new_boxes  

    def _decodeOcrFeat(self, offset, mode="train"):
        ocr_data = None
        if mode == "train":
            ocr_data = self.train_ocr_data[offset]
        else:
            ocr_data = self.val_ocr_data[offset]
        
        new_boxes = np.zeros((ocr_data.shape[0],8), dtype='float32')
        new_feats = np.zeros((ocr_data.shape[0],768), dtype='float32')

        new_boxes = ocr_data[:, :8]
        new_feats = ocr_data[:, 8:]

        return new_boxes, new_feats

    # read ocr feat
    def _loadOcrFeat(self, ocr_path):
        ocr_data = torch.load(ocr_path)

        return ocr_data

    
class VizWizVQAEvaluator:
    def __init__(self, dataset: VizWizVQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        include_oov_score = score / len(quesid2ans)

        no_oov_score = 0.
        no_oov_num = 0
        idx = 0
        for quesid, ans in quesid2ans.items():
            idx += 1
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            true_label = list(label.keys())[0]
            # if idx <= 100:
            #     print(true_label, ans)
            if true_label == "oov":
                continue
            no_oov_num += 1
            if true_label == ans:
                no_oov_score += 1
        no_oov_score = no_oov_score / no_oov_num
        
        return include_oov_score, no_oov_score
    
    def evaluate_ans_type(self, quesid2ans: dict):
        yes_score, other_score, number_score, unanswerable_score = 0., 0., 0., 0.
        yes_num, other_num, number_num, unanswerable_num= 0, 0, 0, 0
        score = 0.

        for quesid, (img_id, ans, ans_type) in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans_type == "unanswerable":
                unanswerable_num += 1
            elif ans_type == "number":
                number_num += 1
            elif ans_type == "yes/no":
                yes_num += 1
            elif ans_type == "other":
                other_num += 1
            if ans in label:
                score += label[ans]
                if ans_type == "unanswerable":
                    unanswerable_score += label[ans]
                elif ans_type == "number":
                    number_score += label[ans]
                elif ans_type == "yes/no":
                    yes_score += label[ans]
                elif ans_type == "other":
                    other_score += label[ans]
        average_score = score / len(quesid2ans)
        yes_score = yes_score / yes_num
        other_score = other_score / other_num
        number_score = number_score / number_num
        unanswerable_score = unanswerable_score / unanswerable_num


        ocr_score = 0.
        ocr_num = 0
        # ocr label evaluation
        for quesid, (img_id, ans, ans_type) in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            labels = datum['label']
            for label in labels.keys():
                if label[:3] == "OCR":
                    ocr_num += 1
                    if ans == label:
                        ocr_score += labels[ans]
                
        ocr_score = ocr_score / ocr_num

        return average_score, (yes_score, other_score, number_score, unanswerable_score, ocr_score)

    
    def evaluate_soft(self, quesid2ans: dict):
        score = 0.
        no_oov_score = 0.
        oov_num = 0
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
                if ans != "oov":
                    no_oov_score += label[ans]
                elif ans == "oov":
                    oov_num += 1
                
        include_oov_score = score / len(quesid2ans)
        no_oov_score = no_oov_score / (len(quesid2ans)-oov_num)

        return include_oov_score, no_oov_score

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, (img_id, ans, answer_type) in quesid2ans.items():
                result.append({
                    'img_id': img_id,
                    'question_id': ques_id,
                    'answer': ans,
                    'answer_type': answer_type
                })
            json.dump(result, f, indent=4, sort_keys=True)


