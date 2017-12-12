from Apply.utils import load_kb_us
from blocking.reconstruction import *
import re

from tools import USED_CAR_DICT


def len_Unknown(labels, DICT):
    count = 0
    for l in labels:
        if l not in DICT.keys():
            count += 1
    return count


def contains(small, big):
    for i in range(len(big)-len(small)+1):
        for j in range(len(small)):
            if big[i+j] != small[j]:
                break
        else:
            return True
            pass
        # 对于一个block:'Math',python 自带的 in 返回是True,我们需要返回为False,因为'Mathematics'返回应该为True
    return False


# 重新定义一个判断函数用在isContain()中,比如'Mathematics and Computers in Simulation',
# 判断year,volume,pages跟判断title,author,journal不一样,前三个要一样,后三个包含就可以，
def my_in(block, l):
    return contains(block.split(), l.split())


# 这个的策略是只要先找到就break程序，然后label就记为找到它所在ＫＢ的类别．
def isContain2(block, KB):
    label = None
    Find = False
    # print('Y')
    for kb in KB['Year']:
        if block == kb:
            label = 'Year'
            Find = True
            return label
    # print('V')
    for kb in KB['Volume']:
        if block == kb:
            label = 'Volume'
            Find = True
            return label
    # print('P')
    for kb in KB['Pages']:
        if block == kb:
            label = 'Pages'
            Find = True
            return label
    # print('A')
    for kb in KB['Author']:
        if block in kb:
            label = 'Author'
            Find = True
            return label
    # print('T')
    for kb in KB['Title']:
        if block in kb:
            label = 'Title'
            Find = True
            return label
    # print('J')
    for kb in KB['Journal']:
        if block in kb:
            label = 'Journal'
            Find = True
            return label
    return label


#记录block出现在所有类中label，以及它在各类中的次数
def isContain3(block, KB):
    block = ' '.join(block)
    # print('block:', block)
    labels = {}
    for kb in KB['Year']:
        if block == kb:
            label = 'Year'
            if label in labels:
                labels[label] += 1
            else:
                labels[label] = 1
    for kb in KB['Volume']:
        if block == kb:
            label = 'Volume'
            if label in labels:
                labels[label] += 1
            else:
                labels[label] = 1
    for kb in KB['Pages']:
        if block == kb:
            label = 'Pages'
            if label in labels:
                labels[label] += 1
            else:
                labels[label] = 1
    for kb in KB['Author']:
        if my_in(block, kb):
            label = 'Author'
            if label in labels:
                labels[label] += 1
            else:
                labels[label] = 1
    for kb in KB['Title']:
        if my_in(block, kb):
            label = 'Title'
            if label in labels:
                labels[label] += 1
            else:
                labels[label] = 1
    for kb in KB['Journal']:
        if my_in(block, kb):
            label = 'Journal'
            if label in labels:
                labels[label] += 1
            else:
                labels[label] = 1
    return labels


# 只进行分块,不计算词频
def isContain4(block, KB):
    block = ' '.join(block)
    find = False
    # print('block:', block)
    for kb in KB['Year']:
        if block == kb:
            find = True
            break
    for kb in KB['Volume']:
        if block == kb:
            find = True
            break
    for kb in KB['Pages']:
        if block == kb:
            find = True
            break
    for kb in KB['Author']:
        if my_in(block, kb):
            find = True
            break
    for kb in KB['Title']:
        if my_in(block, kb):
            find = True
            break
    for kb in KB['Journal']:
        if my_in(block, kb):
            find = True
            break
    return find


def isContain5(block, KB):
    block = ' '.join(block)
    find = False
    for keys, values in KB.items():
        for kb in values:
            if my_in(block, kb):
                # print('kb:', kb)
                find = True
                break
    return find


# 确定一个block的label,输入是一个block中每个word在KB中确定的label,如:block:'Wei-Hsi Hung', label_list:['Author', 'Author']
# 策略:label_list的最后一个label值记为这个block的label
def confirm_label(label_list):
    label = ''
    if label_list:
        for l in label_list:
            if l:
                label = l
    else:
        label = 'Unknow'
    return label


'''
1.最后一项的key
2.如果最后一项有多个key,value最大的key为label
3.year,volume单独的策略,只要在year_kb,volume_kb中出现就确定为year,volume
'''
def confirm_label2(label_list):
    author_confidence = 2
    title_confidence = 2
    journal_confidence = 2
    page_confidence = 2
    year_confidence = 2
    page_confidence = 2
    label = ''
    if label_list[0]:
        for l in label_list:
            if l:
                label = l
                # print(l)
                # confidences = calculate_conficence(l)
                # print(confidences)
    else:
        label = 'Unknown'
    return label


def calculate_conficence(label_list):
    confidences = {}
    for label_dict in label_list:
        # print(label_dict)
        for key in label_dict.keys():
            if key+'_confidence' in confidences:
                confidences[key+'_confidence'] += label_dict[key]
            else:
                confidences[key+'_confidence'] = label_dict[key]
    return confidences


# 属于每个key(label)的block放入到一个list中
def add2dict(my_dict, key, value):
    value = ' '.join(value)
    if key in my_dict.keys():
        my_dict[key].append(value)
    else:
        my_dict[key] = [value]
    return my_dict


# 以block为key,label为值
def add2dict2(my_dict, key, value):
    print('2')


def print_my_dict(d):
    for key in d.keys():
        for value in d[key]:
            print(key+': '+value)
            # print(key + ':'+' '.join(value))


def dict2list(d):
    label_list = []
    block_list = []
    for key in d.keys():
        for value in d[key]:
            # print(key+': '+value)
            label_list.append(key)
            block_list.append(value)
    return label_list, block_list


#
def doBlock2(sample, KB):
    r1 = '\,+'
    # r = '|'.join([r1, r2])
    sample = re.sub(r1, ' ', sample).split()   # convert string to word list
    # print(sample)
    label = ''
    labels = []
    all_blocks = {}
    i = 0
    while i < len(sample):
        block = sample[i:i + 1]
        indice = 0
        for j in range(i+1, len(sample) + 1):
            temp = sample[i:j]
            # print('temp:', temp)

            label =isContain2(' '.join(temp), KB)
            if label is not None:
                labels.append(label)
            # print('label:', label)
            if label and j < len(sample):
                block = temp
            else:
                indice = j - 1
                break
            if j == len(sample):
                print(temp)
                indice = j
                block = temp
                break

        print("labels:", labels)
        # print('label:', confirm_label(labels))
        # print('block:', block)
        all_blocks =add2dict(all_blocks, confirm_label(labels), block)
        if i == indice:
            i += 1
        else:
            i = indice
        labels = []

    return all_blocks


def doBlock3(sample, KB):
    r1 = '\,+'
    # r = '|'.join([r1, r2])
    sample = re.sub(r1, ' ', sample.lower()).split()   # convert string to word list
    print(sample)
    labels = []
    result_blocks = []
    result_labels = []
    i = 0
    while i < len(sample):
        block = sample[i:i + 1]
        indice = 0
        for j in range(i+1, len(sample) + 1):
            temp = sample[i:j]
            print('temp:', temp)

            find =isContain4(temp, KB)   # return dict
            # print('label:', label)
            # if label is not None:
            #     labels.append(label)
            if find and j < len(sample):
                block = temp
            else:
                indice = j - 1
                break
            if j == len(sample):
                # print(temp)
                indice = j
                block = temp
                break
        print('block:', ' '.join(block))
        # print("labels:", labels)
        # print('conficence', calculate_conficence(labels))
        # print('label:', confirm_label2(labels))
        print('=======')
        # print('block:', block)
        # result_labels.append(confirm_label2(labels))
        result_blocks.append(' '.join(block))

        if i == indice:
            i += 1
        else:
            i = indice
        labels = []
    return result_blocks, result_labels


# 先做block过程,再计算block的VF,然后返回归一化的结果和blcok.
# 最后比较threshold来确定最终的anchor
def doBlock4( sample, KB, threshold):
    r1 = '\,+'
    # r2 = '\.+'
    # r = '|'.join([r1, r2])
    sample = re.sub(r1, ' ', sample.lower()).split()   # convert string to word list
    # print(sample)
    result_blocks = []
    i = 0
    while i < len(sample):
        block = sample[i:i + 1]
        indice = 0
        for j in range(i+1, len(sample) + 1):
            temp = sample[i:j]
            # print('temp:', temp)

            find =isContain4(temp, KB)   # return dict
            if find and j < len(sample):
                block = temp
            else:
                indice = j - 1
                break
            if j == len(sample):
                # print(temp)
                indice = j
                block = temp
                break
        result_blocks.append(' '.join(block))
        if i == indice:
            i += 1
        else:
            i = indice
    # 耗时: 2.5s
    print('result_blocks', result_blocks)
    nornalime_vf_list = cal_values_tf2(result_blocks, KB)
    print('nornalime_vf_list:', nornalime_vf_list)
    # 耗时: 2s

    # determine anchors
    anchors = determinte_anchor(nornalime_vf_list, threshold)

    # print('b_end0 %f s' % (b_end2 - start))

    return result_blocks, anchors


# add dict
def doBlock5(sample, KB, threshold):
    r1 = '\,+'
    # r2 = '\.+'
    # r = '|'.join([r1, r2])
    sample = re.sub(r1, ' ', sample).split()   # convert string to word list
    print(sample)
    result_blocks = []
    i = 0
    while i < len(sample):
        block = sample[i:i + 1]
        indice = 0
        for j in range(i+1, len(sample) + 1):
            # print('j:', j)
            temp = sample[i:j]
            print('temp:', temp)

            find =isContain5(temp, KB)   # return dict
            print(find)
            if find and j <= len(sample):
                block = temp
            else:
                indice = j - 1
                break
            if j == len(sample):
                # print(temp)
                indice = j
                block = temp
                break
        # print('indice:', indice)
        # print('i:', i)
        result_blocks.append(' '.join(block))
        if i == indice:
            i += 1
        else:
            i = indice
    print('result_blocks', result_blocks)
    nornalime_vf_list = cal_values_tf3(result_blocks, KB)
    print('nornalime_vf_list:', nornalime_vf_list)

    # determine anchors
    anchors = determinte_anchor(nornalime_vf_list, threshold)

    # print('b_end0 %f s' % (b_end2 - start))

    return result_blocks, anchors


def selectMinKey(a, i, length):
    k = i
    for j in range(i+1, length):
        if a[k] > a[j]:
            k = j
    return k


def selectSort(record, label, block):
    length = len(block)
    indexes = []
    for i in range(len(block)):
        # print(block[i])
        if block[i] not in record:
            return
        index = record.index(block[i])
        indexes.append(index)

    for i in range(length):
        key = selectMinKey(indexes, i, length)
        if key != 1:
            tmp = indexes[i]
            indexes[i] = indexes[key]
            indexes[key] = tmp

            tmp = block[i]
            block[i] = block[key]
            block[key] = tmp

            tmp = label[i]
            label[i] = label[key]
            label[key] = tmp


# calculate frequency of values which contain the specified block in Knowledge Base
def cal_values_tf(block, KB):
    vf_list = []
    for key in KB.keys():
        # print(key)
        # print(len(KB[key]))
        value_count = 0
        for value in KB[key]:
            if my_in(block.lower(), value):
                value_count += 1
        vf = float('%.5f' % (value_count / len(KB[key])))
        vf_list.append(vf)
    return vf_list


# 输入是block的列表
def cal_values_tf2(block_list, KB):
    nornalime_vf_list = []
    for block in block_list:
        # print('block:', block)
        temp_dict = {}
        for key in KB.keys():
            value_count = 0
            for value in KB[key]:
                if my_in(block.lower(), value):
                    value_count += 1
            # print('%s : %d' % (key, value_count))
            if value_count != 0:
                vf = float('%.5f' % (value_count / len(KB[key])))
            else:
                vf = 0.0
            temp_dict[key] = vf
        # print(temp_dict)
        vf_dict = nornalime_vf(temp_dict)
        # print(vf_dict)
        nornalime_vf_list.append(vf_dict)
    return nornalime_vf_list


# FOR USED CAR
def cal_values_tf3(block_list, KB):
    nornalime_vf_list = []
    for block in block_list:
        # print('block:', block)
        temp_dict = {}
        for key in KB.keys():
            value_count = 0
            for value in KB[key]:
                if my_in(block, value):
                    value_count += 1
            # print('%s : %d' % (key, value_count))
            if value_count != 0:
                vf = float('%.5f' % (value_count / len(KB[key])))
            else:
                vf = 0.0
            temp_dict[key] = vf
        # print(temp_dict)
        vf_dict = nornalime_vf(temp_dict)
        # print(vf_dict)
        nornalime_vf_list.append(vf_dict)
    return nornalime_vf_list


# 已知dict的value,求对应的key
def get_keys(my_dict, value):
    for k, v in my_dict.items():
        if v == value:
            return k


def determinte_anchor(label_dict, threshold):
    labels = []
    for t in label_dict:
        value_sum = sum(t.values())
        # print(value_sum)
        if value_sum:
            max_value = max(t.values())
            if max_value >= threshold:
                label = get_keys(t, max_value)
            else:
                label = 'Unknown'
        else:
            label = 'Unknown'
        labels.append(label)
    return labels


# 输入是dict: {'Author': 0.0, 'Volume': 0.0, 'Year': 0.0, 'Pages': 0.0, 'Title': 0.0, 'Journal': 0.00062}
def nornalime_vf(vf_dict):
    vf_sum = sum(vf_dict.values())
    # print(vf_sum)
    for key in vf_dict.keys():
        if vf_sum:
            vf_dict[key] = float('%.5f' % (vf_dict[key] / vf_sum))
        else:
            vf_dict[key] = 0.0
    return vf_dict


# 相邻是不同的anchor标签,从中间分割为两部分,组合成为一个新的block.
def re_block(blocks, anchors):
    revise_label = []
    revise_block = []
    tail = 0
    i = 0
    while i < len(anchors):
        head = anchors[i]
        # print('head: %i, %s' % (i, labels[i]))
        if head != 'Unknown':
            for j in range(len(anchors)-1, i-1, -1): #从后往前找
                # print('%d:%s' % (j, labels[j]))
                if anchors[j] == head:
                    tail = j
                    break
            # print(labels[i:tail+1])
            # print(' '.join(blocks[i:tail+1]))
            revise_block.append(' '.join(blocks[i:tail+1]))
            revise_label.append(anchors[i])
            i = tail + 1
        else:
            # print(blocks[i])
            revise_block.append(blocks[i])
            revise_label.append('Unknown')
            i += 1
    # print(revise_label)
    # print(revise_block)
    return revise_block, revise_label


if __name__ == '__main__':
    KB = load_kb_us()
    line = '2016 Mazda 3 Sedan Touring,$29335,BN Series Touring Sedan 4dr SKYACTIV-Drive 6sp 2.0i [May],0,Soul Red,6 speed Automatic,4 doors 5 seats Sedan,4 cylinder Petrol - Unleaded ULP Aspirated AspiratedL,5.7 (L/100km)'
    line2 = '2015 Audi,$48999,8V Cabriolet 2dr S tronic 7sp 1.4T (CoD) [MY17],4200,Mythos Black,7 speed Automatic,2 doors 4 seats Convertible,4 cylinder Petrol - Premium ULP Turbo Intercooled Turbo IntercooledL,5.1 (L/100km)'
    blocks, anchors = doBlock5(line2, KB, threshold=0.9)
    print(blocks)
    print(anchors)
    re_blocks, re_anchors = re_block(blocks, anchors)
    print(re_blocks)
    print(re_anchors)
