import sys
sys.path.append('..')
from sklearn.cross_validation import KFold
from train_cnn_pos import *
import pandas as pd
from tools import *
import os

names = ['Brand', 'Price', 'Vehicle', 'Odometer', 'Colour', 'Transmission', 'Body', 'Engine', 'Fuel Enconomy']

df = pd.read_csv('data/train_data_split_brand.txt', names=names, header=None).dropna()
df['Odometer'] = df['Odometer'].apply(lambda x: str(x))

Brand = df['Brand'].dropna().values.tolist()
Price = df['Price'].dropna().values.tolist()
Vehicle = df['Vehicle'].dropna().values.tolist()
Odometer = df['Odometer'].dropna().values.tolist()
Colour = df['Colour'].dropna().values.tolist()
Transmission = df['Transmission'].dropna().values.tolist()
Body = df['Body'].dropna().values.tolist()
Engine = df['Engine'].dropna().values.tolist()
Fuel_enconomy = df['Fuel Enconomy'].dropna().values.tolist()
print("Brand:", Brand)
print("Price:", Price)




# all_samples = [b.split() for b in Brand] + [str(b).split() for b in Price] + [str(b).split() for b in Vehicle] + [str(b).split() for b in Odometer]\
#               + [str(b).split() for b in Colour] + [str(b).split() for b in Transmission] + [str(b).split() for b in Body] + [str(b).split() for b in Engine]\
#               + [str(b).split() for b in Fuel_enconomy]
# print(all_samples)

# for b in Fuel_enconomy:
#     print(remove_black_space(sample_pretreatment_disperse_number2(str(b)).split()))

Brand = [remove_black_space(sample_pretreatment_disperse_number2(str(b)).split()) for b in Brand]
print("Brand:", Brand)
Vehicle = [remove_black_space(sample_pretreatment_disperse_number2(str(b)).split()) for b in Vehicle]
Price = [remove_black_space(sample_pretreatment_disperse_number2(str(b)).split()) for b in Price]
Odometer = [remove_black_space(sample_pretreatment_disperse_number2(str(b)).split()) for b in Odometer]
Colour = [remove_black_space(sample_pretreatment_disperse_number2(str(b)).split()) for b in Colour]
Transmission = [remove_black_space(sample_pretreatment_disperse_number2(str(b)).split()) for b in Transmission]
Body = [remove_black_space(sample_pretreatment_disperse_number2(str(b)).split()) for b in Body]
Engine = [remove_black_space(sample_pretreatment_disperse_number2(str(b)).split()) for b in Engine]
Fuel_enconomy = [remove_black_space(sample_pretreatment_disperse_number2(str(b)).split()) for b in Fuel_enconomy]

x_text = Brand + Vehicle + Price + Odometer + Colour + Transmission + Body + Engine + Fuel_enconomy
# print(x_text)

print("Loading data over!")

max_sample_length = max([len(x) for x in x_text])
print("max_record_length:", max_sample_length)

# Parts of Speech Tag
pos_tag_list = makePosFeatures(x_text)
# print(pos_tag_list)
max_pos_length = max([len(x) for x in pos_tag_list])
print("max_pos_length:", max_pos_length)
# pos_tag_list, _ = makePaddedList(pos_tag_list)

#构造/加载 word词典
if os.path.exists('uc_complete_dict.pickle'):
    vocab = load_dict('uc_complete_dict.pickle')
else:
    vocab = makeWordList(x_text)
    save_dict(vocab, 'uc_complete_dict.pickle')
vocab_size = len(vocab)
print("vocab size:", vocab_size)

#构造/加载 POS词典
if os.path.exists('pos.pickle'):
    pos_vocab = load_dict('pos.pickle')
else:
    pos_vocab = makeWordList(pos_tag_list)
    save_dict(pos_vocab, 'pos.pickle')
# pos_vocab = load_dict('pos.pickle')
pos_vocab_size = len(pos_vocab)
print('pos_vocab size:', pos_vocab_size)


#根据词典讲word变成index,然后讲一条记录进行规则化(变成统一长度,用0填充)
print("Preparing pos_train:")
pos_train_raw = map_word2index(pos_tag_list, pos_vocab)
Pos_train = np.array(makePaddedList2(max_sample_length, pos_train_raw, 0))     # shape(13035, 98)
print(Pos_train.shape)


print("Preparing w_train:")
w_train_raw = map_word2index(x_text, vocab)
w_train = np.array(makePaddedList2(max_sample_length, w_train_raw, 0))
# print(w_train)
print("w_train shape:", w_train.shape)
# print(w_train[0])
print("preparing train data over!")


print("Preparing y_train:")
y_train, label_dict_size = build_y_train_used_car_all_attribute(Brand, Price, Vehicle,  Odometer, Colour, Transmission,
                                                                Body, Engine, Fuel_enconomy)
# print(y_train)
print("Preparing y_train over!")

# shuffle here firstly!
print("Shuffle data:")
data_size = len(w_train)
shuffle_indices = np.random.permutation(np.arange(data_size))
p_w_train = Pos_train[shuffle_indices]
s_w_train = w_train[shuffle_indices]
s_y_train = y_train[shuffle_indices]
print(p_w_train.shape)
print(s_w_train.shape)
print(s_y_train.shape)
print('label_dict_size:', label_dict_size)


embedding_dim = 60
pos_emb_dim = 20
# ===================================


print("Start to train:")
print("Initial TrainCNN: ")
train = TrainCNN_POS(
                 vocab_size=vocab_size,
                 embedding_dim=embedding_dim,   # 词向量维度,或者embedding的维度
                 pos_vocab_size=pos_vocab_size,
                 pos_emb_dim=pos_emb_dim,
                 sequence_length=max_sample_length,     # padding之后的句子长度    (27)
                 num_classes=label_dict_size,
                 )
# Split train/test set, use 10_fold cross_validation
print("k_fold train:")
k_fold = KFold(len(s_w_train), n_folds=6)
for train_indices, test_indices in k_fold:
    w_tr, w_te = s_w_train[train_indices], s_w_train[test_indices]
    p_tr, p_te = p_w_train[train_indices], p_w_train[test_indices]
    y_tr, y_te = s_y_train[train_indices], s_y_train[test_indices]
    train.cnn_train_pos(w_tr, w_te, p_tr, p_te, y_tr, y_te)


