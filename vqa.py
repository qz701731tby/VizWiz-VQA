import os
import numpy as np
import collections
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from param import args


from src.modeling import BertLayerNorm, GeLU

from vqa_model import VQAModel
from vqa_vizwiz import VizWizVQADataset, VizWizVQATorchDataset, VizWizVQAEvaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = VizWizVQADataset(splits)
    tset = VizWizVQATorchDataset(dset, args.model)
    evaluator = VizWizVQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)

class VQA:
    def __init__(self):
        # Datasets

        self.valid_tuple = get_data_tuple(args.valid, bs=128, shuffle=False, drop_last=False)
        self.train_tuple = get_data_tuple(args.train, bs=args.batch_size, shuffle=True, drop_last=True)
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=512,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        
        # Model
        self.model = VQAModel(self.train_tuple.dataset.num_answers, args.model)
        # Load pre-trained weights
        if args.load_pretrained is not None:
            self.model.encoder.load(args.load_pretrained)
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from src.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            # for i, (ques_id, feats, boxes, sent, target, answer_type, img_id) in iter_wrapper(enumerate(loader)):
            for i, ( ques_id, feats, boxes, ocr_feats, ocr_boxes, sent, target, answer_type, img_id) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                # feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                feats, boxes, ocr_feats, ocr_boxes, target = feats.cuda(), boxes.cuda(), ocr_feats.cuda(), ocr_boxes.cuda(), target.cuda()
                # logit = self.model(feats, boxes, sent)
                logit = self.model(feats, boxes, ocr_feats, ocr_boxes, sent)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l, ans_type in zip(ques_id, label.cpu().numpy(), answer_type):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = (img_id, ans, ans_type)

            average_score, class_score = evaluator.evaluate_ans_type(quesid2ans)
            # yes_score, other_score, number_score, unanswerable_score = class_score
            yes_score, other_score, number_score, unanswerable_score, ocr_score = class_score
            log_str = "Epoch %d: train average: %0.2f, yes: %0.2f  other: %0.2f  number: %0.2f  unanswerable: %0.2f ocr: %0.2f\n" % \
            (epoch, average_score*100., yes_score*100., other_score*100.0, \
            number_score*100.0, unanswerable_score*100.0, ocr_score*100.0)

            if self.valid_tuple is not None:  # Do Validation
                # include_oov_score, no_oov_score = self.evaluate(eval_tuple)
                average_score, class_score = self.evaluate(eval_tuple)
                # yes_score, other_score, number_score, unanswerable_score = class_score
                yes_score, other_score, number_score, unanswerable_score, ocr_score = class_score
                # if no_oov_score > best_valid:
                #     best_valid = no_oov_score
                #     self.save("BEST")
                if average_score > best_valid:
                    best_valid = average_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid average: %0.2f, yes: %0.2f  other: %0.2f  number: %0.2f  unanswerable: %0.2f ocr: %0.2f\n" % \
                    (epoch, average_score*100., yes_score*100., other_score*100.0, number_score*100.0, unanswerable_score*100.0, ocr_score*100.0) + \
                    "Epoch %d: Best %0.2f\n" % (epoch, best_valid*100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            # ques_id, feats, boxes, sent, _, answer_type, img_ids = datum_tuple   # Avoid seeing ground truth
            ques_id, feats, boxes, ocr_feats, ocr_boxes, sent, _, answer_type, img_ids = datum_tuple
            with torch.no_grad():
                # feats, boxes = feats.cuda(), boxes.cuda()
                feats, boxes, ocr_feats, ocr_boxes = feats.cuda(), boxes.cuda(), ocr_feats.cuda(), ocr_boxes.cuda()
                # logit = self.model(feats, boxes, sent)
                logit = self.model(feats, boxes, ocr_feats, ocr_boxes, sent)
                score, label = logit.max(1)
                for qid, img_id, l, ans_type in zip(ques_id, img_ids, label.cpu().numpy(), answer_type):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = (img_id, ans, ans_type)
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate_ans_type(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s" % path)
        self.model.load_state_dict(state_dict)
        
        
if __name__ == "__main__":
    # Build Class
    vqa = VQA()

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load_trained is not None:
        vqa.load(args.load_trained)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            vqa.predict(
                get_data_tuple(args.test, bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
        elif 'val' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = vqa.evaluate(
                get_data_tuple('val', bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'val_predict.json')
            )
            print(result)
        elif 'train' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = vqa.evaluate(
                get_data_tuple('train', bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'train_predict.json')
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', vqa.train_tuple.dataset.splits)
        if vqa.valid_tuple is not None:
            print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
            # print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        vqa.train(vqa.train_tuple, vqa.valid_tuple)
