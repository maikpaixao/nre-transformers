import torch
from torch import nn, optim
from .base_model import SentenceRE
import sys
sys.path.append("..")
from encoder.base_encoder import BaseEncoder

class SoftmaxNN(SentenceRE):
    def __init__(self, sentence_encoder, num_class, rel2id):
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, item):
        self.eval()
        item = self.sentence_encoder.tokenize(item)
        logits = self.forward(*item)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score
    
    def forward(self, *args):
        rep, semantics, switch = self.sentence_encoder(*args)
        if switch:
            semantics = semantics.squeeze(-1)
            semantics = torch.max(semantics, dim = 2)[0]
            #rep = torch.cat([rep, semantics], 1)
        rep = self.drop(rep)
        logits = self.fc(rep)
        return logits
