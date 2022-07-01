from collections import defaultdict
import numpy as np
import math
from functools import cmp_to_key


class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]

    def find(self, p):
        root = p

        while root != self.parent[root]:
            root = self.parent[root]

        while p != self.parent[p]:
            tmp = self.parent[p]
            self.parent[p] = root
            p = tmp
        return root

    def union(self, p, q):
        p_id = self.find(p)
        q_id = self.find(q)
        if p_id == q_id:
            return 
        self.parent[p_id] = q_id

def get_rect_points(text_boxes):
    text_boxes = np.array(text_boxes)
    x1 = np.min(text_boxes[:, :, 0])
    y1 = np.min(text_boxes[:, :, 1])
    x2 = np.max(text_boxes[:, :, 0])
    y2 = np.max(text_boxes[:, :, 1])

    return [[x1, y1],[x2, y1], [x2, y2],[x1, y2]]

def get_slope(rect):
    x1, x2, _, _ = rect
    if x2[0]-x1[0] == 0:
        return None 

    return (x2[1] - x1[1]) / (x2[0] - x1[0])

def get_char_size(rect):
    x1, x2, x3, x4 = rect
    height = (abs(x4[1]-x1[1]) + abs(x3[1]-x2[1])) / 2
    return height

def get_rect_center(rect):
    x1, _, x3, _ = rect
    return [(x1[0]+x3[0])/2, (x1[1]+x3[1])/2]

def rotate_point(angle, valuex, valuey, pointx, pointy, direction="S"):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    if direction == "S":
        Rotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
        Rotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
    else:
        Rotatex = (valuex-pointx)*math.cos(angle) - (valuey-pointy)*math.sin(angle) + pointx
        Rotatey = (valuex-pointx)*math.sin(angle) + (valuey-pointy)*math.cos(angle) + pointy
    return Rotatex, Rotatey

def rotate_rect(rect, angle, direction="S"):
    rotated_rect = []
    center_x, center_y = get_rect_center(rect)
    for x, y in rect:
        rotated_rect.append(rotate_point(angle, x, y, center_x, center_y, direction))
    
    return rotated_rect

def overlap(min1, max1, min2, max2):
    return max(0, min(max1, max2) - max(min1, min2))

def calc_overlap_for_Yaxis(rect1, rect2):
    '''
        calculate overlap in Y-axis
    '''
    rect1 = np.array(rect1)
    rect2 = np.array(rect2)
    y1_min, y1_max = np.min(rect1[:, 1]), np.max(rect1[:, 1])
    y2_min, y2_max = np.min(rect2[:, 1]), np.max(rect2[:, 1])
    height1, height2 = y1_max-y1_min, y2_max-y2_min
    Yaxis_overlap = overlap(y1_min, y1_max, y2_min, y2_max) / min(height1, height2)

    return Yaxis_overlap

def calc_overlap_for_Xaxis(rect1, rect2):
    '''
        calculate overlap in X-axis
    '''
    rect1 = np.array(rect1)
    rect2 = np.array(rect2)
    x1_min, x1_max = np.min(rect1[:, 0]), np.max(rect1[:, 0])
    x2_min, x2_max = np.min(rect2[:, 0]), np.max(rect2[:, 0])
    width1, width2 = x1_max-x1_min, x2_max-x2_min
    Xaxis_overlap = overlap(x1_min, x1_max, x2_min, x2_max) / min(width1, width2)
    
    return Xaxis_overlap

def judge_two_rects(rect1, rect2, text1, text2, slope_thres, char_thres, overlap_thres, y_dis_thres):
    '''
        judge whether two rects can be merged
        1.倾斜角度
        2.字体大小相似 (the height of char)
        3.垂直或水平的重合度
    '''
#     print("---------------------------")
#     print("text:", text1, text2)
    # the slope of two rects
    slope1 = get_slope(rect1)
    slope2 = get_slope(rect2)
#     print("slope:", slope1, slope2)
    if slope1 == None or slope2 == None:
        return False
    if abs(slope1 - slope2) > slope_thres:
        return False
    
    # the char size
    char_size1 = get_char_size(rect1)
    char_size2 = get_char_size(rect2)
#     print("char_size:", char_size1, char_size2)
    if abs(char_size1 - char_size2)/max(abs(char_size1), abs(char_size2)) > char_thres:
        return False
    
    # overlap
    average_k = (get_slope(rect1)+get_slope(rect2)) / 2
    angle = np.arctan(average_k)
    rotated_rect1 = rotate_rect(rect1, angle, direction="S")
    rotated_rect2 = rotate_rect(rect2, angle, direction="S")
    x_overlap = calc_overlap_for_Xaxis(rotated_rect1, rotated_rect2)
    y_overlap = calc_overlap_for_Yaxis(rotated_rect1, rotated_rect2)
#     print("dist:", dist, (1+dis_thres) * side_len)
    if x_overlap < overlap_thres and y_overlap < overlap_thres:
        return False
    
    center1 = get_rect_center(rect1)
    center2 = get_rect_center(rect2)
#     print("y_dist:", (center1[1]-center2[1])/min(char_size1, char_size2))
    if abs(center1[1]-center2[1])/((char_size1+char_size2)/2) > y_dis_thres:
        return False

    return True

def rect_cmp(rect1, rect2):
    rect1, rect2 = rect1[0], rect2[0]
    char_size = min(get_char_size(rect1), get_char_size(rect2))
    center1 = get_rect_center(rect1)
    center2 = get_rect_center(rect2)
    if abs(center1[1]-center2[1])/char_size > 0.3:
        return center1[1] < center2[1]
    
    return center1[0] < center2[0]    
    
def merge_rects(rects, texts):
    '''
        merge a list of rects into one
    '''
    # calculate average angle
    average_k = 0
    for rect in rects:
        average_k += get_slope(rect)
    average_k /= len(rects)
    angle = np.arctan(average_k)
    rotated_rects = []
    for rect in rects:
        rotated_rects.append(rotate_rect(rect, angle, direction="S"))
    sorted_rects = sorted(list(zip(rotated_rects, texts)), key=cmp_to_key(rect_cmp),reverse=True)
    new_text = " ".join([rect[1] for rect in sorted_rects])
    new_rect = get_rect_points(rects)
    new_rect = rotate_rect(new_rect, angle, direction="N")

    return new_rect, new_text
    
def boxes_connect(rects, texts, slope_thres, char_thres, dis_thres, y_dis_thres):
    n = len(rects)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i+1, n):
            if judge_two_rects(rects[i], rects[j], texts[i], texts[j], slope_thres, char_thres, dis_thres, y_dis_thres):
#                 print(i, j)
#                 print(texts[i], texts[j])
                uf.union(i, j)

    connected_idx = defaultdict(list)
    for i in range(n):
        connected_idx[uf.find(i)].append(i)
    
    connected_boxes = list(connected_idx.values())
#     print("connected_boxes:", connected_boxes)
    new_rects, new_texts = [], []
    for conn in connected_boxes:
        new_rect, new_text = merge_rects([rects[idx] for idx in conn], [texts[idx] for idx in conn])
        new_rects.append(new_rect)
        new_texts.append(new_text)
    
    return new_rects, new_texts

def point_rot90(x, y, drt, size):
    H, W = size
    if drt == 0:
        return x, y
    if drt == 1:
        return H-y, x
    if drt == 2:
        return W-x, H-y
    if drt == 3:
        return y, W-x

def rect_rot90(rects, drt, size):
    '''
        rotate rect to original direction
    '''
    new_rects = []
    for rect in rects:
        new_rect = []
        for x, y in rect:
            new_rect.append(point_rot90(x, y, drt, size))
        new_rects.append(new_rect)
    
    return new_rects