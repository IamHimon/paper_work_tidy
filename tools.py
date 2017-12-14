import re
import nltk
import pickle
import numpy as np
LABEL_DICT = {'Title': 0, 'Author': 1, 'Journal': 2, 'Year': 3, 'Volume': 4, 'Pages': 5}
USED_CAR_DICT = {'Brand': 0, 'Price': 1, 'Vehicle': 2, 'Odometer': 3, 'Color': 4, 'Transmission': 5, 'Body': 6, 'Engine': 7, 'Fuel_enconomy': 8}
label = ['Vehicle', 'Price', 'Odometer', 'Colour', 'Transmission', 'Body', 'Engine', 'Fuel Enconomy']



def remove_black_space(a):
    c = []
    for i in a:
        if i != ' ':
            c.append(i)
    return c


# 输入处理trick, 讲数字成分用空格分离开, '2015 Audi'-> '2 0 1 5 Audi'
def sample_pretreatment_disperse_number2(sample):
    add_length = 0
    for m in re.finditer(r'\d+', sample):
        # print('start:',m.start())
        # print('end:', m.end())
        # print('add_length:', add_length)
        sample = replace_by_position(sample, m.start()+add_length, m.end()+add_length)
        add_length += (m.end() - m.start()) + 1
    #     print(sample)
    # print(sample)
    return sample


# 用空格链接一段字符, 比如: '2015' -> ' 2 0 1 5 '
def replace_by_position(str, start, end):
    seg_str = str[start:end]
    temp = ' ' + ' '.join([c for c in seg_str]) + ' '
    str = str[0:start] + temp + str[end:len(str)]
    return str


# 构造Pos特征
def makePosFeatures(sent_contents):
    """
    :param sent_contents:
    :return:sent_contents中每个sentence都构建一个list,存放sentence中每个word的标注信息.
    """
    pos_tag_list = []
    for sent in sent_contents:
        # print(sent)

        pos_tag = nltk.pos_tag(sent)
        # print(pos_tag)
        pos_tag = list(zip(*pos_tag))[1]    # 拆开pos_tag
        # print(pos_tag)
        pos_tag_list.append(pos_tag)
    return pos_tag_list


def load_dict(dict_path):
    with open(dict_path, 'rb') as handle:
        b = pickle.load(handle)
    return b


def map_word2index(x_text, word_dict):
    # print("map word to index:")
    w_train = []
    temp = []
    for x in x_text:
        # print(x)
        for w in x:
            if w in word_dict:
                temp.append(word_dict[w])
            else:
                temp.append(0)
        w_train.append(temp)
        temp = []
    return w_train


# longer-trim, shorter-padding,用pad_symbol填充
def makePaddedList2(maxl, sent_contents, pad_symbol):
    T = []
    for sent in sent_contents:
        t = []
        lenth = len(sent)
        if lenth < maxl:
            for i in range(lenth):
                t.append(sent[i])
            for i in range(lenth, maxl):
                t.append(pad_symbol)
        else:
            for i in range(maxl):
                t.append(sent[i])
        T.append(t)
    return T


def build_y_train_used_car_all_attribute(Brand, Price, Vehicle,  Odometer, Colour, Transmission, Body, Engine, Fuel_enconomy):
    # print("Building label dict:")
    Brand_labels = [0 for i in range(len(Brand))]
    Price_labels = [1 for i in range(len(Price))]
    Vehicle_labels = [2 for i in range(len(Vehicle))]
    Odometer_labels = [3 for i in range(len(Odometer))]
    Colour_labels = [4 for i in range(len(Colour))]
    Transmission_labels = [5 for i in range(len(Transmission))]
    Body_labels = [6 for i in range(len(Body))]
    Engine_labels = [7 for i in range(len(Engine))]
    Fuel_enconomy_labels = [8 for i in range(len(Fuel_enconomy))]

    y_t = Brand_labels + Price_labels + Vehicle_labels + Odometer_labels + Colour_labels + Transmission_labels + \
          Body_labels + Engine_labels + Fuel_enconomy_labels
    label_dict_size = len(USED_CAR_DICT)

    y_train = np.zeros((len(y_t), label_dict_size))
    for i in range(len(y_t)):
        y_train[i][y_t[i]] = 1
    # print("Preparing y_train over!")
    return y_train, label_dict_size


def makeWordList(sent_list):
    """
    :param sent_list:
    :return:返回一个字典,{'word1':index1,'word2':index2,....},index从1开始
    """
    wf = {}
    for sent in sent_list:      # 构造字典wf,键是单个word,值是出现的次数
        for w in sent:
            if w in wf:
                wf[w] += 1
            else:
                wf[w] = 0
    wl = {}
    i = 0
    wl['unkown'] = 0
    for w, f in wf.items():     # 构造字典wl,键是单个word, 值是下标,从1开始
        # print(w, ' ', f)
        i += 1
        wl[w] = i
    return wl


# save dict
def save_dict(word_dict, name):
    with open(name, 'wb') as handle:
        pickle.dump(word_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

