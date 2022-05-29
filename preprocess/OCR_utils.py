import json
import re
from collections import Counter

#编辑距离
def editDist(str1, str2):
    # str1: answer; str2: ocr;
    str1 = str1.lower()
    str2 = str2.lower()
    editTable = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            editTable[i][j] = min(editTable[i - 1][j] + 1, editTable[i][j - 1] + 1, editTable[i - 1][j - 1] + d)
    return editTable[len(str1)][len(str2)] / max(len(str1), len(str2))

#单词对齐
def alignScore(str1, str2):
    words1 = str1.split()
    words2 = str2.split(" ")
    aligns = 0
    pos = 0
    for word in words1:
        minEditDist = 10000
        minPos = pos
        for i in range(pos, len(words2)):
            curEditDist = editDist(word, words2[i])
            if curEditDist < minEditDist:
                minEditDist = curEditDist
                minPos = i
        if minEditDist <= 0.2: ## This threshold controls how much edit dist can be considered as a matching pair;
            pos = minPos + 1
            aligns += 1
    return aligns / min(len(words1), len(words2))


def alignScore1(str1, str2):
    words1 = str1.split()
    words2 = str2.split(" ")
    aligns = 0
    pos = 0
    for word in words1:
        minEditDist = 10000
        minPos = pos
        for i in range(pos, len(words2)):
            curEditDist = editDist(word, words2[i])
            if curEditDist < minEditDist:
                minEditDist = curEditDist
                minPos = i
        if minEditDist == 0: ## This threshold controls how much edit dist can be considered as a matching pair;
            pos = minPos + 1
            aligns += 1
    return aligns / len(words1)

#去标点符号
def question_processing(q):
    # Remove punctuations; all possible punctuations on the keyboard;
    punctuation_dict = {"!": '', "@": '', "#": '', "$": '', "%": '', "^": '', "&": '', "*": '', "(": '', ")": '', "_": '', "+": '', "=": '', "[": '', "]": '', "{": '', "}": '', ":": '', ";": '', "'": '', '"': '', "<": '', ">": '', ",": '', ".": '', "?": '', "/": '', "|": '', "\\": '', "`": '', "~": '',}
    rep = punctuation_dict
    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
    q = pattern.sub(lambda m: rep[re.escape(m.group(0))], q)
    return q

#带OCR的label和OCR文件
def getocr(path1, path2):
    val_data = json.load(open(path1,'r',encoding='UTF-8'))
    ocr_data = json.load(open(path2, 'r', encoding = 'UTF-8'))
    #替换OCR
    #OCR结果
    ocrdic = {}
    for item in val_data:
        if(item["answer_type"] == "unanswerable"):
            continue
        #当前图片的OCR结果（位置+识别语句）
        data = ocr_data[item["image"]]
        #若OCR的结果为空，直接保存
        if(len(data) == 0):
            ocrdic[item["image"]] = data
            continue
        for ocrit in data:
            ocrit.append(ocrit[1])
        #统计OCR（识别语句）
        ocrans = set()
        sent = []
        for it in item["answers"]:
            sent.append(it['answer'])
            for ocrit in data:
                #通过edit distance筛选
                if len(it["answer"]) < 2*len(ocrit[1]) and len(ocrit[1]) < 2*len(it["answer"]):
                    ocrProcessed = question_processing(ocrit[1])
                    editDistance = editDist(it["answer"], ocrProcessed)
                    alignScoreVal = alignScore(it["answer"], ocrProcessed)
                    #edit distance
                    if editDistance <= 0.3 or (editDistance > 0.3 and alignScoreVal > 0.6):
                        #将当前answer替换为OCR结果
                        it["answer"] = ocrit[1]
                        ocrans.add(ocrit[1])
        num = Counter(sent)
        dictionary = sorted(num.items(), key = lambda x:x[1], reverse=True)
        for ocrdata in data:
            for ans in dictionary:
                if len(ans[0]) < 2*len(ocrdata[2]) and len(ocrdata[2]) < 2*len(ans[0]):
                    ocrProcessed = question_processing(ocrdata[2])
                    editDistance = editDist(ans[0], ocrProcessed)
                    alignScoreVal = alignScore(ans[0], ocrProcessed)
                    #edit distance
                    if editDistance <= 0.3 or (editDistance > 0.3 and alignScoreVal > 0.6):
                        #将当前OCR-result替换为结果最多的answer
                        ocrdata[2] = ans[0]
                        break
        #对当前图片OCR结果，根据面积从大到小排序
        areadic = sorted(data, key=lambda s:(s[0][2][0]-s[0][0][0])*(s[0][2][1]-s[0][0][1]), reverse=True)
        #存需要保留的OCR结果
        ocrreslis = []
        #若有超过三个OCR可匹配，根据面积大小选出三个
        if len(ocrans) >= 3:
            for it in areadic:
                if it[1] in ocrans:
                    ocrreslis.append(it)
            ocrreslis = ocrreslis[0:2]
        #若不到三个OCR可匹配
        else:
            #若data本身不到三个，全部加入
            if len(data) <= 3:
                ocrreslis = [i for i in data]
            #若data本身大于三个，匹配数量不到三个
            else:
                addlen = 3 - len(ocrans)
                for dataitem in data:
                    if dataitem[1] in ocrans:
                        ocrreslis.append(dataitem)
                        ocrans.remove(dataitem[1])
                        areadic.remove(dataitem)
                for i in range(addlen):
                    ocrreslis.append(areadic[i])
        #排序data，按照先左，后下的顺序
        lefttop = sorted(ocrreslis, key = lambda s:(s[0][0][0], s[0][0][1]))
        ocrresdic = {}
        for i in range(len(lefttop)):
            ocrstr = "OCR" + str(i+1)
            ocrresdic[lefttop[i][1]] = ocrstr
        for it in item["answers"]:
            if it["answer"] in ocrresdic.keys():
                flag = it["answer"]
                it["answer"] = ocrresdic[flag]
        ocrdic[item["image"]] = lefttop
    ansdata = []
    for item in val_data:
        if item['answer_type'] != "unanswerable":
            ansdata.append(item)
    return ocrdic, ansdata