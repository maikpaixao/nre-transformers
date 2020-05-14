# coding:utf-8
import torch
import numpy as np
import json
sys.path.append("..")
from encoder import bert_encoder
from model import softmax_nn
from framework import sentence_re

ckpt = 'ckpt/semeval_bert_softmax.pth.tar'
rel2id = json.load(open('../benchmark/semeval/semeval_rel2id.json'))
sentence_encoder = bert_encoder.BERTEncoder(
    max_length=80, pretrain_path='../benchmark/bert-base-uncased')
model = softmax_nn.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
framework = sentence_re.SentenceRE(train_path='../benchmark/semeval/train.txt',
                                    val_path='../benchmark/semeval/val.txt',
                                    test_path='../benchmark/semeval/test.txt',
                                    model=model, ckpt=ckpt, batch_size=64,
                                    max_epoch=10, lr=3e-5, opt='adam')
# Train
framework.train_model(metric='micro_f1')

# Test
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)

f = open("output_bert_path_chunk.txt","w+")
f.write('Precision: ' + str(result['micro_p']) + '\n')
f.write('Recall: ' + str(result['micro_r']) + '\n')
f.write('Score: ' + str(result['micro_f1']) + '\n')
f.close()
