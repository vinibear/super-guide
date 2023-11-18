import torch
import torch.nn as nn
import math
import numpy as np
from utils import wash_text, pad_and_mask, trans_qkv, re_trans_qkv
from utils import Word2Vectors
import torch.nn.functional as F


"""from utils import *
from model import *
t = read_binary_file("C:/Users/10435/Desktop/abstract.txt")
emb = Word_Embedding(5000, 10, 'model/tt.bin', 'model/tt.bin')
washed = [wash_text(t)]
emb.refresh_vocabulary(washed)
out1, mask1 = emb([t, t])
emb2 = PositionalEncoding(10)
print(out1)
print("PPPPPPPPPPPP")
out2=emb2(out1.transpose(0, 1)).transpose(0, 1)
print(out2)
print(out2*mask1)"""


class PositionalEncoding(nn.Module):
    """
    用法示范：
    emb2=PositionalEncoding(d_model=10,max_len=100)
    emb2.eval()
    emb2(a.transpose(0,1)).transpose(0,1)-a-emb2(torch.zeros(100,1,10)).transpose(0,1)<torch.ones(1,100,10)/10e6
    """

    def __init__(self, d_model=100, dropout=0.1, max_len=5000):
        # d_model是一个点的特征维度，max_len是最长序列长度
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # pe的维度是（5000，512）
        pe = torch.zeros(max_len, d_model)
        # position是一个5000行1列的tensor
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term是一个256长度的一维tensor
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 最终的pe是一个torch.Size([5000, 1, 512])的维度
        self.pe = pe

    def forward(self, x):
        '''
        x: [seq_len,batch_size,d_model]
        '''

        x = x + self.pe[:x.size(0), :]
        # return x
        return self.dropout(x)


class Word_Embedding(nn.Module):
    """
    用法示范：

    t=read_binary_file("C:/Users/10435/Desktop/abstract.txt")
    emb=Word_Embedding(10,3,'model/tt.bin','model/tt.bin')
    washed=[wash_text(t)]
    emb.refresh_vocabulary(washed)
    emb([t])

    """

    def __init__(self, max_length=5000, vector_size=100, pretrained_model="",
                 save_path=""):
        super(Word_Embedding, self).__init__()
        self.wd2vc = Word2Vectors(
            train=False,  pretrained_model=pretrained_model,
            vector_size=vector_size)
        self.save_path = save_path
        self.max_length = max_length
        self.vector_size = vector_size

    def forward(self, batched_text):

        lengths = torch.zeros(
            (len(batched_text), self.max_length, self.vector_size))
        good_texts = torch.zeros(
            (len(batched_text), self.max_length, self.vector_size))
        # print(batched_text)
        for i, text in enumerate(batched_text):
            # print(f"{i}    PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP")
            # print(text)

            washed_text = wash_text(
                text, process='lemma', keep_number=False,
                output_article=True)
            # print("PPPPPP")
            # print(washed_text)

            numpy_array = np.array([self.wd2vc(word) for word in washed_text])
            # 将NumPy数组转换为张量
            translated_text = torch.from_numpy(numpy_array)

            # print("PPPssdds")
            # print(translated_text)
            good_texts[i], lengths[i] = pad_and_mask(
                translated_text, self.max_length)
        return good_texts, lengths

    def refresh_vocabulary(self, washed_text):
        self.wd2vc.refresh_vocabulary(washed_text, save_path=self.save_path)


class DotProductAttention(nn.Module):
    def __init__(self, dropout=0.1,):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V):
        # [batch_size*num_head,max_length,num_hiddens]
        d = Q.shape[-1]
        qk = self.dropout(torch.bmm(Q, K.transpose(1, 2))/math.sqrt(d))
        out = self.softmax(torch.bmm(qk, V))
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, num_hiddens, num_vecters, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.attention = DotProductAttention(dropout)
        self.num_heads = num_heads
        self.num_hiddens = num_hiddens
        self.num_vecters = num_vecters
        self.Wq = nn.Linear(num_vecters, num_heads*num_hiddens)
        self.Wk = nn.Linear(num_vecters, num_heads*num_hiddens)
        self.Wv = nn.Linear(num_vecters, num_heads*num_hiddens)
        self.out = nn.Linear(num_heads*num_hiddens, num_vecters)

    def forward(self, Sq, Sk, Sv):
        Q = self.Wq(Sq)
        K = self.Wq(Sk)
        V = self.Wq(Sv)

        Q1 = trans_qkv(Q, self.num_heads)
        K1 = trans_qkv(K, self.num_heads)
        V1 = trans_qkv(V, self.num_heads)

        X1 = self.attention(Q1, K1, V1)
        X = re_trans_qkv(X1, self.num_heads)
        return self.out(X)


class Add_and_Norm(nn.Module):
    def __init__(self, num_vecters, dropout=0.1):
        super(Add_and_Norm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(num_vecters)

    def forward(self, X, Y):
        return self.norm(self.dropout(Y) + X)


class PositionWiseFFN(nn.Module):
    def __init__(self, input_size, ffn_hidden_size, output_size):
        super(PositionWiseFFN, self).__init__()
        self.ffn_1 = nn.Linear(input_size, ffn_hidden_size)
        self.ffn_2 = nn.Linear(ffn_hidden_size, output_size)

    def forward(self, X):
        return self.ffn_2(F.relu(self.ffn_1(X)))


class Encode_Block(nn.Module):
    def __init__(self, num_heads, num_hiddens, num_vecters, ffn_hidden_size,
                 dropout=0.1):
        super(Encode_Block, self).__init__()
        self.add_norm1 = Add_and_Norm(num_vecters, dropout)
        self.add_norm2 = Add_and_Norm(num_vecters, dropout)
        self.attention = MultiHeadAttention(
            num_heads, num_hiddens, num_vecters, dropout)
        self.ffn = PositionWiseFFN(num_vecters, ffn_hidden_size, num_vecters)

    def forward(self, X):
        X1 = self.attention(X, X, X)
        X1 = self.add_norm1(X, X1)
        X2 = self.ffn(X1)
        out = self.add_norm1(X1, X2)
        return out


class Encoder(nn.Module):
    def __init__(self, max_length, num_heads, num_hiddens,
                 num_vecters, ffn_hidden_size, num_Encoder_layers,
                 pretrained_model='vocabularies/vocabulary.bin',
                 save_path="vocabularies/vocabulary",
                 dropout=0.1, device="cuda"):
        super(Encoder, self).__init__()
        self.word_emb = Word_Embedding(max_length=max_length,
                                       vector_size=num_vecters,
                                       pretrained_model=pretrained_model,
                                       save_path=save_path)

        self.pos_emb = PositionalEncoding(
            d_model=num_vecters, dropout=dropout, max_len=max_length)

        self.layers = nn.ModuleList()

        for i in range(num_Encoder_layers):
            self.layers.append(Encode_Block(num_heads, num_hiddens,
                                            num_vecters, ffn_hidden_size,
                                            dropout))
        self.layers.to(device)
        self.device = device

    def forward(self, texts_list, refresh_vocabulary=False):
        # t = read_binary_file("C:/Users/10435/Desktop/abstract.txt")
        # emb = Word_Embedding(5000, 10, 'model/tt.bin', 'model/tt.bin')
        # washed = [wash_text(t)]
        if refresh_vocabulary:
            print("refresh_vocabulary")
            for t in texts_list:
                washed = [wash_text(t)]
                self.word_emb.refresh_vocabulary(washed)
        emb1, mask1 = self.word_emb(texts_list)
        # emb2 = PositionalEncoding(10)
        # print(out1)
        # print("PPPPPPPPPPPP")
        emb2 = self.pos_emb(emb1.transpose(0, 1)).transpose(0, 1)
        # print(out2)
        # print(out2*mask1)
        X = emb2*mask1
        X = X.to(self.device)
        for layer in self.layers:
            X = layer(X)

        return X


class Decoder(nn.Module):
    def __init__(self, num_vecters, num_hiddens, num_out, device="cuda"):
        super(Decoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.rnn = nn.RNN(input_size=num_vecters,
                          hidden_size=num_hiddens).to(device)
        self.output = nn.Linear(num_hiddens, num_out).to(device)
        self.softmax = nn.Softmax(dim=-1).to(device)
        self.device = device

    def begin_state(self, batch_size):
        return torch.zeros((1, batch_size, self.num_hiddens))

    def forward(self, X):
        X = X.transpose(0, 1).contiguous().to(self.device)
        state = self.begin_state(X.shape[1]).to(self.device)

        # print(X.shape,state.shape)
        for x in X:
            # print(x.shape)
            _, state = self.rnn(x.unsqueeze(0), state)

        state = state.squeeze(0)
        out = self.output(state)
        return self.softmax(out)


class Model(nn.Module):
    def __init__(self, max_length=500,
                 num_heads=8,
                 att_hiddens_size=8,
                 num_vecters=10,
                 ffn_hidden_size=20,
                 num_Encoder_layers=3,
                 rnn_hiddens_size=100,
                 output_size=10,
                 path_to_save_vocabulary="vocabularies/vocabulary",
                 pretrained_vocabulary='vocabularies/vocabulary.bin',
                 encoder_dropout=0.1,
                 device="cuda"):
        super(Model, self).__init__()
        self.encoder = Encoder(max_length=max_length,
                               num_heads=num_heads,
                               num_hiddens=att_hiddens_size,
                               num_vecters=num_vecters,
                               ffn_hidden_size=ffn_hidden_size,
                               num_Encoder_layers=num_Encoder_layers,
                               pretrained_model=pretrained_vocabulary,
                               save_path=path_to_save_vocabulary,
                               dropout=encoder_dropout, device=device)

        self.decoder = Decoder(num_vecters=num_vecters,
                               num_hiddens=rnn_hiddens_size,
                               num_out=output_size,
                               device=device)

    def forward(self, texts, refresh=False):
        mid = self.encoder(texts, refresh)
        out = self.decoder(mid)
        return out
