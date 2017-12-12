import pandas as pd
import tensorflow as tf
from tools import *


# 从文件构造Knowledge Base
def load_kb_us():
    # filename = 'cars3.txt'
    # records = load_car_data(filename)
    names = ['Brand', 'Price', 'Vehicle', 'Odometer', 'Colour', 'Transmission', 'Body', 'Engine', 'Fuel Enconomy']
    # df = pd.DataFrame(records, columns=names)

    df = pd.read_csv('../data/train_data_split_brand.txt', names=names)

    # print(df)
    Brand = df['Brand'].dropna().values.tolist()
    Price = df['Price'].dropna().values.tolist()
    Vehicle = df['Vehicle'].dropna().values.tolist()
    Odometer = df['Odometer'].dropna().apply(lambda x: str(int(x))).values.tolist()
    Colour = df['Colour'].dropna().values.tolist()
    Transmission = df['Transmission'].dropna().values.tolist()
    Body = df['Body'].dropna().values.tolist()
    Engine = df['Engine'].dropna().values.tolist()
    # Fuel_enconomy = df['Fuel Enconomy'].dropna().values.tolist()

    Fuel_enconomy = data_aug_fe()

    KB = {'Brand': Brand, 'Price': Price, 'Vehicle': Vehicle, 'Odometer': Odometer, 'Colour': Colour, 'Transmission': Transmission,
          'Body': Body, 'Engine': Engine, 'Fuel Enconomy': Fuel_enconomy}

    kb_write = open('../data/uc_knowledge_base.txt', 'w+')
    for k, v in KB.items():
        kb_write.write(k + '\n')
        write_set(kb_write, v)

    # print(KB)
    return KB


# 构造Odometer
def data_aug_fe():
    fe_list = []
    fe = 0.0
    for i in range(150):
        fe += 0.1
        # print(str('%.1f' % (fe)) + ' (L/100km)')
        fe_list.append(str('%.1f' % (fe)) + ' (L/100km)')
    return fe_list


def write_set(fw, my_set):
    for i in my_set:
        fw.write(i + '\n')



# block = np.arrange(len(blocks))
def greddy_predictions(loss, block):
    predictions = loss.argmax(1)
    score = loss.max(1)

    # print('prediction:', predictions)
    # print('score:', score)
    # print('sum score:', np.sum(score))
    if len(loss[0]) < len(LABEL_DICT):
        return predictions, score

    result = []
    copy_predictions = predictions.copy()
    # print(copy_predictions)
    # print(loss)
    # print(block)
    l = 0
    for ind, p in enumerate(predictions):
        max_temp = -10000
        max_index = 0
        # print(predictions)
        # print('p:', str(p))
        index = np.where(copy_predictions == p)
        # print(index[0])
        if len(index[0]) > 1:
            max_index_index = 0
            for i in index[0]:
                if max_temp < loss[i][p]:
                    max_temp = loss[i][p]
                    max_index_index = i
            # print('max_temp', max_temp)
            # print('index:', ind)
            # score[ind] = max_temp
            # print('max_index:', str(max_index))
            # print('rest label:')
            # print(make_rest_label(copy_predictions, block))
            for i in index[0]:
                if i != max_index_index:
                    # print(i)
                    rest_label_index = make_rest_label(copy_predictions, block)[0]
                    copy_predictions[i] = rest_label_index
                    # print(loss[rest_label_index][p])
        #     max_index = predictions[max_index_index]
        # else:
        #     max_index = predictions[index[0][0]]
        # print('max_index:', max_index)
        # print(loss[l][max_index])
        # score[l] = loss[l][max_index]
        # l += 1
        # print(copy_predictions)
        # print('max_index:', max_index)
        # print(loss[rest_label_index][p])
    if (np.sort(copy_predictions) == block).all():
        result = [str(p) for p in copy_predictions]
    # print(predictions)
    # print(copy_predictions)
    revise_score = revise_score_list(predictions, copy_predictions, loss)
    # print(score)
    # print(revise_score)
    return result, revise_score


def make_rest_label(predictions, block):
    rest_label = []
    for b in block:
        if b not in predictions:
            rest_label.append(b)
    return rest_label


def revise_score_list(predictions, copy_predictions, loss):
    score = []
    for i in range(len(predictions)):
        if predictions[i] != copy_predictions[i]:
            score.append(loss[i][copy_predictions[i]])
        else:
            score.append(loss[i][predictions[i]])
    return np.array(score)


def max_tensor_score(temp_list, sess):
    max_s = tf.constant(0.0)
    result = None
    for t in temp_list:
        # print(sess.run(t[1]))
        if sess.run(tf.less(max_s, t[1])):
            # print(sess.run(max_s))
            max_s = t[1]
            result = t[0]
    return result


def len_Unknown2(labels, DICT):
    count = 0
    for l in labels:
        if l not in DICT.keys():
            count += 1
    return count