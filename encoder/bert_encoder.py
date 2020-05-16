import logging
import math, logging
import torch
import torch.nn as nn
import pickle
from transformers import BertModel, BertTokenizer
from .base_encoder import BaseEncoder
from .utils import Utils

class BERTEncoder(nn.Module):
    def __init__(self, token2id, max_length, pretrain_path, blank_padding=True, mask_entity=False, word2vec = None,
                        e_position = False, e_path = False, e_chunks = False, e_semantics = False):

        super().__init__()
        self.max_length = max_length
        self.token2id = token2id
        self.num_token = len(token2id)
        self.blank_padding = blank_padding
        self.hidden_size = 768
        self.mask_entity = mask_entity
        self.e_position = e_position
        self.e_chunks = e_chunks
        self.e_path = e_path
        self.e_semantics = e_semantics

        self.pos1_embedding = nn.Embedding(2 * max_length, 5, padding_idx=0)
        self.pos2_embedding = nn.Embedding(2 * max_length, 5, padding_idx=0)
        self.path_embedding = nn.Embedding(2 * max_length, 40, padding_idx=0)

        if word2vec is None:
            self.word_size = word_size
        else:
            self.word_size = word2vec.shape[-1]

        if not '[UNK]' in self.token2id:
            self.token2id['[UNK]'] = len(self.token2id)
            self.num_token += 1
        if not '[PAD]' in self.token2id:
            self.token2id['[PAD]'] = len(self.token2id)
            self.num_token += 1
        
        self.word_embedding = nn.Embedding(self.num_token, self.word_size)

        if word2vec is not None:
            logging.info("Initializing word embedding with word2vec.")
            word2vec = torch.from_numpy(word2vec)
            if self.num_token == len(word2vec) + 2:
                unk = torch.randn(1, self.word_size) / math.sqrt(self.word_size)
                blk = torch.zeros(1, self.word_size)
                self.word_embedding.weight.data.copy_(torch.cat([word2vec, unk, blk], 0))
            else:
                self.word_embedding.weight.data.copy_(word2vec)

        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)

        if e_position:
            self.hidden_size = self.hidden_size + 10
        if e_path:
            self.hidden_size = self.hidden_size + 40
        if e_chunks:
            self.hidden_size = self.hidden_size + 50
        if e_semantics:
            self.hidden_size = self.hidden_size + 50

    def forward(self, token, att_mask, pos1, pos2, path, chunks, semantics):
        pos1 = self.pos1_embedding(pos1)
        pos2 = self.pos2_embedding(pos2)
        path = self.path_embedding(path)
        chunks = self.word_embedding(chunks)
        #_, x = self.bert(token, attention_mask=att_mask)

        if self.e_position:
            x = torch.cat([x, pos1, pos2], 1)
        elif self.e_path:
            x = torch.cat([x, path], 1)
        elif self.e_chunks:
            x = torch.cat([x, chunks], 1)
        elif self.e_position and self.e_path:
            x = torch.cat([x, pos1, pos2, path], 1)
        elif self.e_position and self.e_chunks:
            x = torch.cat([x, pos1, pos2, chunk], 1)
        elif self.e_path and self.e_chunks:
            x = torch.cat([x, path, chunks], 1)
        elif self.e_position and self.e_path and self.e_chunks:
            x = torch.cat([x, pos1, pos2, path, chunks], 1)

        if self.e_semantics:
            semantics = self.word_embedding(semantics)

        return x

    def tokenize(self, item):
        utils = Utils(cnn=False)
        
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True

        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']
        ses1 = item['semantics']['ses1']
        ses2 = item['semantics']['ses2']
        semantic = ses1 + ses2
        chunks = item['chunks']
        path = item['path']['embed']

        if not is_token:
            pos_min = pos_head
            pos_max = pos_tail
            if pos_head[0] > pos_tail[0]:
                pos_min = pos_tail
                pos_max = pos_head
                rev = True
            else:
                rev = False
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
            if self.mask_entity:
                ent0 = ['[unused4]']
                ent1 = ['[unused5]']
                if rev:
                    ent0 = ['[unused5]']
                    ent1 = ['[unused4]']
            pos_head = [len(sent0), len(sent0) + len(ent0)]
            pos_tail = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
            if rev:
                pos_tail = [len(sent0), len(sent0) + len(ent0)]
                pos_head = [
                    len(sent0) + len(ent0) + len(sent1),
                    len(sent0) + len(ent0) + len(sent1) + len(ent1)
                ]
            tokens = sent0 + ent0 + sent1 + ent1 + sent2
        else:
            tokens = sentence

        re_tokens = ['[CLS]']
        cur_pos = 0
        pos1 = 0
        pos2 = 0
        for token in tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                pos1 = len(re_tokens)
                re_tokens.append('[unused0]')
            if cur_pos == pos_tail[0]:
                pos2 = len(re_tokens)
                re_tokens.append('[unused1]')
            if is_token:
                re_tokens += self.tokenizer.tokenize(token)
            else:
                re_tokens.append(token)
            if cur_pos == pos_head[1] - 1:
                re_tokens.append('[unused2]')
            if cur_pos == pos_tail[1] - 1:
                re_tokens.append('[unused3]')
            cur_pos += 1

        re_tokens.append('[SEP]')

        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)
        
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        indexed_path = self.tokenizer.convert_tokens_to_ids(utils.formatr(path))
        indexed_chunks = self.tokenizer.convert_tokens_to_ids(utils.formatr(chunks))
        indexed_semantic = self.tokenizer.convert_tokens_to_ids(utils.formatr(semantic))

        avai_len = len(indexed_tokens)

        pos1 = torch.tensor(pos1).long().unsqueeze(0)
        pos2 = torch.tensor(pos2).long().unsqueeze(0)

        
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            while len(indexed_path) < self.max_length:
                indexed_path.append(0)
            while len(indexed_chunks) < self.max_length:
                indexed_chunks.append(0)
            while len(indexed_semantic) < self.max_length:
                indexed_semantic.append(0)

            indexed_tokens = indexed_tokens[:self.max_length]
            indexed_path = indexed_path[:self.max_length]
            indexed_chunks = indexed_chunks[:self.max_length]
            indexed_semantic = indexed_semantic[:self.max_length]

        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)
        indexed_path = torch.tensor(indexed_path).long().unsqueeze(0)
        indexed_chunks = torch.tensor(indexed_chunks).long().unsqueeze(0)
        indexed_semantic = torch.tensor(indexed_semantic).long().unsqueeze(0)

        att_mask = torch.zeros(indexed_tokens.size()).long()
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2, indexed_path, indexed_chunks, indexed_semantic
