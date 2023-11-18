import re
import nltk
import numpy as np
import torch
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def best_num_validation(y_hat, y0, best_num=3):
    y0 = y0.to('cpu')
    good = 0
    all = y0.shape[0]
    arrays = np.array(y_hat.to('cpu'))
    arrays = [np.argpartition(array, -best_num)[-best_num:]
              for array in arrays]
    for i in range(y0.shape[0]):
        for x in arrays[i]:
            if x == y0[i]:
                good += 1
    # print(y_hat,arrays)
    return good, all


def make_vocabulary(information_dir, pretrained_vocabubary, save_path,
                    vector_size):

    with open(information_dir, 'r') as train_information:
        lines = train_information.readlines()
    texts = []
    for line in lines:
        adress = line.split()[1]
        # print(adress)
        text = read_file(adress)
        texts.append(wash_text(text))

    Word2Vectors(train=True, train_text=texts,
                 pretrained_model=pretrained_vocabubary,
                 vector_size=vector_size,
                 save_path=save_path)


def read_binary_file(dir):
    with open(dir, 'rb') as f:
        lines = bytes.decode(f.read())
    return lines


def read_file(dir):
    with open(dir, 'r', encoding='utf-8') as f:
        print(dir)
        lines = f.read()
        # print(lines)
    return lines


def wash_text(text, process='lemma', keep_number=False,
              output_article=True):
    """
    输入：
        text：字符串
        process：词形归一(lemma)，词干提取(stem)
        keep_letter_and_number:是否保留数字
        output_article:输出的list是一句句的还是一整篇
    输出：
        一篇洗好的文章，list格式
    """
    text = text.lower()
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    text = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", " ", text)
    lines = text.split('.')
    output = []
    for line in lines:

        if keep_number:
            line = re.sub(
                u"([^\u0041-\u005a\u0061-\u007a\u0030-\u0039\u0020-\u0020])",
                " ", line)
        else:
            line = re.sub(
                u"([^\u0041-\u005a\u0061-\u007a\u0020-\u0020])", " ", line)
        # splited_lines=lines.split('.')
        tokens = nltk.word_tokenize(line)

        # 标词性
        # pos_tags = nltk.pos_tag(tokens)

        # 去stop word
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [
            token for token in tokens if token.lower() not in stop_words]

        if process == 'lemma':  # 词形归一
            good_words = [lemmatizer.lemmatize(
                word, pos='v') for word in filtered_tokens]
        elif process == 'stem':  # 词干提取
            good_words = [stemmer.stem(word) for word in filtered_tokens]
        else:
            print("error at process_text\n\n\n\n*****************************")
            good_words = None
        if not len(good_words) == 0:
            output.append(good_words)
    if output_article:
        output = [word for line in output for word in line]

    return output


def pad_and_mask(X, max_length):
    len1 = len(X)
    wid1 = len(X[0])
    if (len1 >= max_length):
        X = X[:max_length]
        mask = torch.ones((max_length, wid1))
    else:
        X = torch.cat([X, torch.zeros((max_length-len1), wid1)], 0)
        mask = torch.cat(
            [torch.ones((len1, wid1)),
             torch.zeros((max_length-len1, wid1))], 0)
    return X, mask


def pad_and_length(X, max_length):
    len1 = len(X)
    wid1 = len(X[0])
    if (len1 >= max_length):
        X = X[:max_length]
        length = max_length
    else:
        X = torch.cat([X, torch.zeros((max_length-len1), wid1)], 0)
        length = len1
    return X, length


def trans_qkv(X, num_heads):
    X = X.view(X.shape[0], X.shape[1], num_heads, -1)
    X = X.transpose(2, 1).contiguous()
    return X.view(-1, X.shape[2], X.shape[3])


def re_trans_qkv(X, num_heads):
    X = X.view(-1, num_heads, X.shape[1], X.shape[2])
    X = X.transpose(2, 1).contiguous()
    return X.view(X.shape[0], X.shape[1], -1)


class Word2Vectors:
    def __init__(self, train=False, train_text=[], pretrained_model="",
                 vector_size=512, save_path='model/word2vec'):
        """
        train_text是一个list of list
        pretrained_model 预训练模型的地址
        vector_size embedding的向量维度
        save_path 训练完存在哪，如：'model/3'
        """
        self.save_path = save_path
        if train:
            assert (not len(train_text) == 0)
            if len(pretrained_model) == 0:
                model = Word2Vec(sentences=train_text,
                                 vector_size=vector_size,
                                 window=5,
                                 min_count=1)
                model.save(save_path+'.bin')
                model.wv.save_word2vec_format(save_path+".dict")
                self.model = model
            else:
                model = Word2Vec.load(pretrained_model)
                model.build_vocab(train_text, update=True)
                model.train(train_text, epochs=50,
                            total_examples=model.corpus_count)
                model.save(save_path+'.bin')
                model.wv.save_word2vec_format(save_path+".dict")
                self.model = model
        else:
            assert (not len(pretrained_model) == 0)
            model = Word2Vec.load(pretrained_model)
            self.model = model

    def __call__(self, word):
        """
        输入字符串，返回一个词向量，例子：
            a('computer')
        """
        return self.model.wv[word]

    def refresh_vocabulary(self, train_text=[], save_path=''):
        """
        train_text是一个list of list
        save_path 举例：'model/tt'
        """
        assert (not len(train_text) == 0)
        if len(save_path) == 0:
            save_path = self.save_path
        self.model.build_vocab(train_text, update=True)
        self.model.train(train_text, epochs=50,
                         total_examples=self.model.corpus_count)
        self.model.save(save_path+'.bin')
        self.model.wv.save_word2vec_format(save_path+".dict")


class MyIterator:
    def __init__(self, train_information_dir, batch_size):
        self.texts = []
        self.labels = []
        with open(train_information_dir, 'r') as train_information:
            lines = train_information.readlines()
        for line in lines:
            label, adress = line.split()
            self.labels.append(int(label))
            text = read_binary_file(adress)
            self.texts.append(text)
        num = len(self.labels)//batch_size*batch_size
        self.labels = self.labels[:num]
        self.texts = self.texts[:num]

        self.batch_size = batch_size
        # print(len(self.labels))

    def return_iter(self):
        # print(self.texts)
        # print(self.labels )
        for i in range(0, len(self.labels), self.batch_size):
            yield self.texts[i:i+self.batch_size], self.labels[i:i+self.batch_size]
            # print(1)

    def __iter__(self):
        return self.return_iter()
