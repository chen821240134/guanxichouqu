

# 系统相关
import os
import codecs
import csv
import json

#第三方
import jieba
# 自定义
from BruceNRE.utils import load_csv, load_json, ensure_dir, sava_pkl
from BruceNRE.config import config
from BruceNRE.vocab import Vocab

# 判断数据集中是否存在关系字段
def exist_realtion(file, file_type):#file='data/origin\\train.csv'
    with codecs.open(file, encoding='utf-8') as f:
        if file_type == 'csv':
            f = csv.DictReader(f)#DictReader会将第一行的内容（类标题）作为key值，第二行开始才是数据内容。即图中的csv文件有2列7行数据，第一列的key值为id，第二列的key值为class：
        for line in f:
            keys = list(line.keys())#['sentence', 'relation', 'head', 'head_type', 'head_offset', 'tail', 'tail_type', 'tail_offset']
            try:
                num = keys.index('relation')
            except:
                num = -1
            return int(num)
# 分词
def split_sentences(raw_data):
    new_data_list = []
    jieba.add_word("HEAD")#add_word(word, freq=None, tag=None)向词典中添加一个词。freq 和 tag 可以省略，freq 默认为一个计算值
    jieba.add_word("TAIL")
    for data in raw_data:
        head, tail = data[2], data[5]#head:'网络小说' tail:网站  文中的实体
        new_sent = data[0].replace(data[1], 'HEAD', 1)#将实体网络小说替换成HEAD
        new_sent = new_sent.replace(data[4], 'TAIL', 1)#str.replace(old, new[, max])old -- 将被替换的子字符串。new -- 新字符串，用于替换old子字符串。max -- 可选字符串, 替换不超过 max 次
        new_sent = jieba.lcut(new_sent)#分词 lcut 方法直接返回 list，cut 方法返回一个 可迭代的 generator 默认是精确模式
        head_pos, tail_pos = new_sent.index("HEAD"), new_sent.index("TAIL")#返回HEAD和TAIL出现的位置索引
        new_sent[head_pos] = head#是词级别，和数据集中的字级别位置不同
        new_sent[tail_pos] = tail
        data.append(new_sent)#用head_type和tail_type替换文中的head和tail文字
        data.append([head_pos, tail_pos])#在原句加入处理好的data和替换索引['《逐风行》是百度文学旗下纵横中文网签约作家清水秋风创作的一部东方玄幻小说，小说已于2014-04-28正式发布', '逐风行', '网络小说', '1', '纵横中文网', '网站', '12', ['《', '网络小说', '》', '是', '百度', '文学', '旗下', '网站', '签约', '作家', '清水', '秋风', '创作', '的', '一部', '东方', '玄幻', '小说', '，', '小说', '已于', '2014', '-', '04', '-', '28', '正式', '发布'], [1, 7]]
        new_data_list.append(data)
    return new_data_list

# 构建词典
def bulid_vocab(raw_data, out_path):
    if config.word_segment:#默认为TRUE
        vocab = Vocab('word')
        for data in raw_data:
            vocab.add_sentences(data[-2])#data[-2]是data中的分词部分
    else:
        vocab = Vocab('char')
        for data in raw_data:
            vocab.add_sentences(data[0])
    vocab.trim(config.min_freq)#此时vocab有18649个词需要过滤一些min_freq = 过滤掉低频词 id2word 共有6618个词 0为pad 1为unk

    ensure_dir(out_path)#保存out文件夹

    vocab_path = os.path.join(out_path, 'vocab.pkl')#os.path.join()函数：连接两个或更多的路径名组件out_path=‘data/out’
    vocab_txt = os.path.join(out_path, 'vocab.txt')
    sava_pkl(vocab_path, vocab, 'vocab')#保存词典字典类型

    with codecs.open(vocab_txt, 'w', encoding='utf-8') as f:#保存词 没有统计次数
        f.write(os.linesep.join([word for word in vocab.word2idx.keys()]))#keys() 方法返回 view 对象。这个视图对象包含列表形式的字典键。
    return vocab, vocab_path

# 获取位置编码特征
def get_pos_feature(sent_len, entity_pos, entity_len, pos_limit):
    """
    :param sent_len:
    :param entity_pos:
    :param entity_len:
    :param pos_limit:
    :return:
    """
    left = list(range(-entity_pos, 0))#第一部分左侧为负数第二部分为0×实体长度，其实是一，因为词结构，右半部分补齐长度
    middle = [0] * entity_len
    right = list(range(1, sent_len - entity_pos - entity_len + 1))
    pos = left + middle + right
    for i, p in enumerate(pos):
        if p > pos_limit:
            pos[i]  = pos_limit
        if p < -pos_limit:
            pos[i] = -pos_limit
    pos = [p + pos_limit + 1 for p in pos]#保证取值在1到50之间
    return pos

def get_mask_feature(entities_pos, sen_len):
    """
    获取mask编码
    :param entities_pos:
    :param sen_len:
    :return:
    """
    left = [1] * (entities_pos[0] + 1)
    middle = [2] * (entities_pos[1] - entities_pos[0] - 1)
    right = [3] * (sen_len - entities_pos[1])
    return left + middle + right

def bulid_data(raw_data, vocab):
    sents = []
    head_pos = []
    tail_pos = []
    mask_pos = []

    if vocab.name == 'word':
        for data in raw_data:
            sent = [vocab.word2idx.get(w,1) for w in data[-2]]#将词转换为数字
            pos = list(range(len(sent)))#将数字从小到大排序
            head, tail = int(data[-1][0]), int(data[-1][-1])#获取被替换词的位置索引
            entities_pos = [head, tail] if tail > head else [tail, head]#实体索引[1,7]
            head_p = get_pos_feature(len(sent), head, 1, config.pos_limit)#28 1 1 50处理后得到的长度与sent一致   pos_limit = 50
            tail_p = get_pos_feature(len(sent), tail, 1, config.pos_limit)#28 7 1 50
            mask_p = get_mask_feature(entities_pos, len(sent))
            sents.append(sent)
            head_pos.append(head_p)
            tail_pos.append(tail_p)
            mask_pos.append(mask_p)
    else:
        for data in raw_data:
            sent = [vocab.word2idx.get(w, 1) for w in data[0]]
            head, tail = int(data[3]), int(data[6])
            head_len, tail_len = len(data[1]), len(data[4])
            entities_pos = [head, tail] if tail > head else [tail, head]
            head_p = get_pos_feature(len(sent), head, head_len, config.pos_limit)
            tail_p = get_pos_feature(len(sent), tail, tail_len, config.pos_limit)
            mask_p = get_mask_feature(entities_pos, len(sent))
            sents.append(sent)
            head_pos.append(head_p)
            tail_pos.append(tail_p)
            mask_pos.append(mask_p)
    return sents, head_pos, tail_pos, mask_pos

def relation_tokenize(relations, file):
    """

    :param relations:
    :param file:
    :return:
    """
    relations_list = []
    relations_dict = {}
    out = []
    with codecs.open(file, encoding='utf-8') as f:#file='data/origin\\relation.txt'
        for line in f:
            relations_list.append(line.strip())#relations_list['国籍', '祖籍', '导演', '出生地', '主持人', '所在城市', '所属专辑', '连载网站', '出品公司', '毕业院校']
    for i, rel in enumerate(relations_list):
        relations_dict[rel] = i
    for rel in relations:
        out.append(relations_dict[rel])
    return out


# 数据预处理
def process(data_path, out_path, file_type):
    print("*****数据预处理开始*****")
    file_type = file_type.lower()#csv
    assert file_type in ['csv', 'json']#断言可以在条件不满足程序运行的情况下直接返回错误

    print("*****加载原始数据*****")

    train_fp = os.path.join(data_path, 'train.' + file_type)#读取文件，join连接起来
    test_fp = os.path.join(data_path, 'test.' + file_type)
    relation_fp = os.path.join(data_path, 'relation.txt')

    relation_place = exist_realtion(train_fp, file_type)#relation_place=1 # 判断数据集中是否存在关系字段 感觉这个没啥用，因为给出的数据集明显有relation这一列

    if relation_place > -1:
        if file_type == 'csv':
            train_raw_data = load_csv(train_fp)#一共4000条 读取文件的value 一共8列[['《逐风行》是百度文学旗下纵横中文网签约作家清水秋风创作的一部东方玄幻小说，小说已于2014-04-28正式发布', '连载网站', '逐风行', '网络小说', '1', '纵横中文网', '网站', '12'],
            test_raw_data = load_csv(test_fp)#一共1000条
        else:
            train_raw_data = load_json(train_fp)
            test_raw_data = load_json(test_fp)

        train_relations = []
        test_relations = []

        for data in train_raw_data:
            train_relations.append(data.pop(relation_place))#取出第一个位置  pop()函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。.
        for data in test_raw_data:#注意 train_raw_data就没有relation这一列了
            test_relations.append(data.pop(relation_place))

        if config.is_chinese and config.word_segment:#默认都是TRUE
            train_raw_data = split_sentences(train_raw_data)#返回给新的数据，在原句加入处理好的data和替换索引
            test_raw_data = split_sentences(test_raw_data)

        print("构建词典")
        vocab, vocab_path = bulid_vocab(train_raw_data, out_path)
        #vocab构建好的字典 6618个字
        print("构建train模型数据")
        train_sents, train_head_pos, train_tail_pos, train_mask_pos = bulid_data(train_raw_data, vocab);
        #词转换成数字索引
        print("构建test模型数据")
        test_sents, test_head_pos, test_tail_pos, test_mask_pos = bulid_data(test_raw_data, vocab)

        print("构建关系数据")
        train_relations_token = relation_tokenize(train_relations, relation_fp)#将关系转换为数字4000个
        test_relations_token = relation_tokenize(test_relations, relation_fp)

        ensure_dir(out_path)
        train_data = list(
            zip(train_sents, train_head_pos, train_tail_pos, train_mask_pos, train_relations_token)
        )
        test_data = list(
            zip(test_sents, test_head_pos, test_tail_pos, test_mask_pos, test_relations_token)
        )

        train_data_path = os.path.join(out_path, 'train.pkl')#构建好的训练集
        test_data_path = os.path.join(out_path, 'test.pkl')

        sava_pkl(train_data_path, train_data, 'train data')
        sava_pkl(test_data_path, test_data, 'test data')

        print("*****数据预处理完成*****")



