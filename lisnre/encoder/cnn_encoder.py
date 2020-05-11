import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_encoder import BaseEncoder
from .base_encoder import GloveEncoder

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
                 mask_entity=False):
        """
        Args:
            token2id: dictionary of token->idx mapping
            max_length: max length of sentence, used for postion embedding
            hidden_size: hidden size
            word_size: size of word embedding
            position_size: size of position embedding
            blank_padding: padding for CNN
            word2vec: pretrained word2vec numpy
            kernel_size: kernel_size size for CNN
            padding_size: padding_size for CNN
        """
        # Hyperparameters
        super(CNNEncoder, self).__init__(token2id, max_length, hidden_size, word_size, position_size, blank_padding, word2vec, mask_entity=mask_entity)
        self.drop = nn.Dropout(dropout)
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.act = activation_function

        self.conv = nn.Conv1d(self.input_size, self.hidden_size, self.kernel_size, padding=self.padding_size)
        self.pool = nn.MaxPool1d(self.max_length)

    def forward(self, token, pos1, pos2, xs, ys):
        """
        Args:
            token: (B, L), index of tokens
            pos1: (B, L), relative position to head entity
            pos2: (B, L), relative position to tail entity
        Return:
            (B, EMBED), representations for sentences
        """
        # Check size of tensors
        if len(token.size()) != 2 or token.size() != pos1.size() or token.size() != pos2.size():
            raise Exception("Size of token, pos1 ans pos2 should be (B, L)")
        x = torch.cat([self.word_embedding(token),
                       self.pos1_embedding(pos1),
                       self.pos2_embedding(pos2),
                       self.word_embedding(xs),
                       self.word_embedding(ys)], 4) # (B, L, EMBED)
        print(x.shape)
        x = x.transpose(1, 2) # (B, EMBED, L)
        x = self.act(self.conv(x)) # (B, H, L)
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
        """
        Args:
            token2id: dictionary of token->idx mapping
            max_length: max length of sentence, used for postion embedding
            hidden_size: hidden size
            word_size: size of word embedding
            position_size: size of position embedding
            blank_padding: padding for CNN
            word2vec: pretrained word2vec numpy
            kernel_size: kernel_size size for CNN
            padding_size: padding_size for CNN
        """
        # Hyperparameters
        super(WEEncoder, self).__init__(token2id, max_length, hidden_size, word_size, position_size, blank_padding, word2vec, mask_entity=mask_entity)
        self.drop = nn.Dropout(dropout)
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.act = activation_function

        self.conv = nn.Conv1d(self.input_size, self.hidden_size, self.kernel_size, padding=self.padding_size)
        self.pool = nn.MaxPool1d(self.max_length)

    def forward(self, token):
        """
        Args:
            token: (B, L), index of tokens
            pos1: (B, L), relative position to head entity
            pos2: (B, L), relative position to tail entity
        Return:
            (B, EMBED), representations for sentences
        """
        # Check size of tensors
        if len(token.size()) != 2 or token.size() != pos1.size() or token.size() != pos2.size():
            raise Exception("Size of token, pos1 ans pos2 should be (B, L)")
        x = torch.cat([self.word_embedding(token),
                       self.pos1_embedding(pos1),
                       self.pos2_embedding(pos2)], 2) # (B, L, EMBED)
        x = x.transpose(1, 2) # (B, EMBED, L)
        x = self.act(self.conv(x)) # (B, H, L)
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        torch.save(tensor, 'file.pt')
        return x

    def tokenize(self, item):
        return super().tokenize(item)

class SCNNEncoder(BaseEncoder):

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

        super(SCNNEncoder, self).__init__(token2id, max_length, hidden_size, word_size, position_size, blank_padding, word2vec, mask_entity=mask_entity)
        self.drop = nn.Dropout(dropout)
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.act = activation_function
        self.conv = nn.Conv1d(self.input_size, self.hidden_size, self.kernel_size, padding=self.padding_size)
        self.pool = nn.AvgPool1d(self.kernel_size, stride=2, padding=self.padding_size)

    def forward(self, token):
        # Check size of tensors
        x = self.word_embedding(token).unsqueeze(2)
        x = self.act(self.conv(x)) # (B, H, L) (230, 1)
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return x

    def tokenize(self, item):
        sentence = item['token']
        tokens = sentence

        if self.blank_padding:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, self.max_length, self.token2id['[PAD]'], self.token2id['[UNK]'])
        else:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, unk_id = self.token2id['[UNK]'])

        #if self.blank_padding:
        #    indexed_tokens = indexed_tokens[:self.max_length]

        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)

        return indexed_tokens
