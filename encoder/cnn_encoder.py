import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_encoder import BaseEncoder
import sys
sys.path.append("..")
from module.nn.cnn import CNN
from module.pool.max_pool import MaxPool

class CNNEncoder(BaseEncoder):
    def __init__(self,
                 token2id,
                 max_length=128,
                 hidden_size=230,
                 word_size=50,
                 position_size=5,
                 blank_padding=True,
                 word2vec=None,
                 kernel_size=3,
                 padding_size=1,
                 dropout=0,
                 activation_function=F.relu,
                 mask_entity = False,
                 e_position = False,
                 e_path = False,
                 e_chunks = False,
                 e_semantics = False):

        super(CNNEncoder, self).__init__(token2id, max_length, hidden_size, word_size, position_size, blank_padding, word2vec, mask_entity=mask_entity)
        self.e_position = e_position
        self.e_path = e_path
        self.e_chunks = e_chunks
        self.e_semantics = e_semantics
        self.input_size = 50
        self.drop = nn.Dropout(dropout)
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.act = activation_function
        self.conv = nn.Conv1d(self.input_size, self.hidden_size, self.kernel_size, padding=self.padding_size)
        self.pool = nn.MaxPool1d(self.max_length)
        if e_position:
            self.input_size = self.input_size + 10
        if e_path:
            self.input_size = self.input_size + 50
        if e_chunks:
            self.input_size = self.input_size + 50
        if e_semantics:
            self.input_size = self.input_size + 100

    def forward(self, token, pos1, pos2, path, chunks, ses1, ses2):
        x = self.word_embedding(token).unsqueeze(2)
        '''
        if e_position:
            x = torch.cat([self.word_embedding(token), 
                       self.pos1_embedding(pos1)], 2)
        if e_path:
            x = torch.cat([self.word_embedding(token), 
                       self.pos1_embedding(pos1)], 2)
        if e_chunks:
            x = torch.cat([self.word_embedding(token), 
                       self.pos1_embedding(pos1)], 2)
        if e_semantics:
            x = torch.cat([self.word_embedding(token), 
                       self.pos1_embedding(pos1)], 2)

        x = torch.cat([self.word_embedding(token), 
                       self.pos1_embedding(pos1)], 2) # (B, L, EMBED)
        '''
        #x = x.transpose(1, 2) # (B, EMBED, L)
        x = self.act(self.conv(x))
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return x

    def tokenize(self, item):
        return super().tokenize(item)

class POSCNNEncoder(BaseEncoder):

    def __init__(self,
                 token2id,
                 max_length=128,
                 hidden_size=230,
                 word_size=50,
                 position_size=0,
                 blank_padding=True,
                 word2vec=None,
                 kernel_size=1,
                 padding_size=1,
                 dropout=0,
                 activation_function=F.relu,
                 mask_entity=False):

        super(POSCNNEncoder, self).__init__(token2id, max_length, hidden_size, word_size, position_size, blank_padding, word2vec, mask_entity=mask_entity)
        self.drop = nn.Dropout(dropout)
        self.input_size = 50 + self.position_size*2
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.act = activation_function

        self.conv = nn.Conv1d(self.input_size, self.hidden_size, self.kernel_size, padding=self.padding_size)
        self.pool = nn.MaxPool1d(self.max_length)

    def forward(self, token, pos1, pos2, path, chunks, ses1, ses2):
        if len(token.size()) != 2 or token.size() != pos1.size() or token.size() != pos2.size():# or token.size() != xs.size() or token.size() != ys.size():
            raise Exception("Size of token, pos1 ans pos2 should be (B, L)")
        x = torch.cat([self.word_embedding(token),
                       self.pos1_embedding(pos1),
                       self.pos2_embedding(pos2)], 2) 

        x = x.transpose(1, 2)
        x = self.act(self.conv(x))
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return x

    def tokenize(self, item):
        return super().tokenize(item)

class PATHCNNEncoder(BaseEncoder):

    def __init__(self,
                 token2id,
                 max_length=128,
                 hidden_size=230,
                 word_size=50,
                 position_size=5,
                 blank_padding=True,
                 word2vec=None,
                 kernel_size=3,
                 padding_size=1,
                 dropout=0,
                 activation_function=F.relu,
                 mask_entity=False):

        super(PATHCNNEncoder, self).__init__(token2id, max_length, hidden_size, word_size, position_size, blank_padding, word2vec, mask_entity=mask_entity)
        self.drop = nn.Dropout(dropout)
        self.input_size = 50*2
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.act = activation_function
        self.conv = nn.Conv1d(self.input_size, self.hidden_size, self.kernel_size, padding=self.padding_size)
        self.pool = nn.MaxPool1d(self.max_length)

    def forward(self, token, pos1, pos2, path, chunks, ses1, ses2):
        if len(token.size()) != 2 or token.size() != pos1.size() or token.size() != pos2.size():# or token.size() != xs.size() or token.size() != ys.size():
            raise Exception("Size of token, pos1 ans pos2 should be (B, L)")
        x = torch.cat([self.word_embedding(token),
                       self.word_embedding(path)], 2) 

        x = x.transpose(1, 2)
        x = self.act(self.conv(x))
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return x

    def tokenize(self, item):
        return super().tokenize(item)

class CHUNKCNNEncoder(BaseEncoder):
    def __init__(self,
                 token2id,
                 max_length=128,
                 hidden_size=230,
                 word_size=50,
                 position_size=5,
                 blank_padding=True,
                 word2vec=None,
                 kernel_size=3,
                 padding_size=1,
                 dropout=0,
                 activation_function=F.relu,
                 mask_entity=False):

        super(CHUNKCNNEncoder, self).__init__(token2id, max_length, hidden_size, word_size, position_size, blank_padding, word2vec, mask_entity=mask_entity)
        self.drop = nn.Dropout(dropout)
        self.input_size = 50*2
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.act = activation_function

        self.conv = nn.Conv1d(self.input_size, self.hidden_size, self.kernel_size, padding=self.padding_size)
        self.pool = nn.MaxPool1d(self.max_length)

    def forward(self, token, pos1, pos2, path, chunks, ses1, ses2):
        if len(token.size()) != 2 or token.size() != pos1.size() or token.size() != pos2.size():# or token.size() != xs.size() or token.size() != ys.size():
            raise Exception("Size of token, pos1 ans pos2 should be (B, L)")
        x = torch.cat([self.word_embedding(token),
                       self.word_embedding(chunks)], 2)

        x = x.transpose(1, 2)
        x = self.act(self.conv(x))
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return x

    def tokenize(self, item):
        return super().tokenize(item)

class SEMCNNEncoder(BaseEncoder):
    def __init__(self,
                 token2id,
                 max_length=128,
                 hidden_size=230,
                 word_size=50,
                 position_size=5,
                 blank_padding=True,
                 word2vec=None,
                 kernel_size=3,
                 padding_size=1,
                 dropout=0,
                 activation_function=F.relu,
                 mask_entity=False):

        super(SEMCNNEncoder, self).__init__(token2id, max_length, hidden_size, word_size, position_size, blank_padding, word2vec, mask_entity=mask_entity)
        self.drop = nn.Dropout(dropout)
        self.input_size = 50*3
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.act = activation_function

        self.conv = nn.Conv1d(self.input_size, self.hidden_size, self.kernel_size, padding=self.padding_size)
        self.pool = nn.MaxPool1d(self.max_length)

    def forward(self, token, pos1, pos2, path, chunks, ses1, ses2):
        if len(token.size()) != 2 or token.size() != pos1.size() or token.size() != pos2.size():# or token.size() != xs.size() or token.size() != ys.size():
            raise Exception("Size of token, pos1 ans pos2 should be (B, L)")
        x = torch.cat([self.word_embedding(token),
                       self.word_embedding(ses1),
                       self.word_embedding(ses2)], 2) 

        x = x.transpose(1, 2) 
        x = self.act(self.conv(x))
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return x

    def tokenize(self, item):
        return super().tokenize(item)


###################################################################################
###################### FIRST ROUND FINISHES HERE ##################################
###################################################################################
