import logging
import torch
import torch.nn as nn
import pickle
from transformers import BertModel, BertTokenizer
from .base_encoder import BaseEncoder
from .utils import Utils

class BERTEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False,
                        e_position = False, e_path = False, e_chunks = False, e_semantics = False):
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768
        self.mask_entity = mask_entity
        self.e_position = e_position
        self.e_chunks = e_chunks
        self.e_path = e_path
        self.e_semantics = e_semantics

        self.pos1_embedding = nn.Embedding(2 * max_length, 5, padding_idx=0)
        self.pos2_embedding = nn.Embedding(2 * max_length, 5, padding_idx=0)

        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)

    def forward(self, token, att_mask, pos1, pos2, path):
        _, x = self.bert(token, attention_mask=att_mask)
        '''
        if self.e_position:
            pos1 = self.pos1_embedding(pos1)
            pos2 = self.pos2_embedding(pos2)
        if self.e_path:
            self.input_size = self.input_size + 50
        if self.e_chunks:
            self.input_size = self.input_size + 50
        if self.e_semantics:
            self.input_size = self.input_size + 100
        '''
        #x = torch.cat([x, pos1, pos2], 1)
        return x, 0, self.e_semantics

    def tokenize(self, item):
        utils = Utils(cnn=False)
        # Sentence -> token
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

        # Token -> index
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

        # Position -> index
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)
        

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        indexed_path = self.tokenizer.convert_tokens_to_ids(utils.formatr(path))
        indexed_chunks = self.tokenizer.convert_tokens_to_ids(utils.formatr(chunks))
        indexed_semantic = self.tokenizer.convert_tokens_to_ids(utils.formatr(semantic))

        avai_len = len(indexed_tokens)

        # Position
        pos1 = torch.tensor(pos1).long().unsqueeze(0)
        pos2 = torch.tensor(pos2).long().unsqueeze(0)

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            while len(indexed_path) < self.max_length:
                indexed_path.append(0)
                #indexed_chunks.append(0)
                #indexed_semantic.append(0)
            indexed_tokens = indexed_tokens[:self.max_length]
            indexed_path = indexed_path[:self.max_length]
            #indexed_chunks = indexed_chunks[:self.max_length]
            #indexed_semantic = indexed_semantic[:self.max_length]

        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)
        indexed_path = torch.tensor(indexed_path).long().unsqueeze(0)
        #indexed_chunks = torch.tensor(indexed_chunks).long().unsqueeze(0)
        #indexed_semantic = torch.tensor(indexed_semantic).long().unsqueeze(0)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2, indexed_path
