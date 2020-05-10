# coding:utf-8
import torch
import numpy as np
import json
import opennre
from opennre import encoder, model, framework
from encoder import cnn_encoder

ckpt = 'ckpt/semeval_cnn_softmax.pth.tar'
wordi2d = json.load(open('../pretrain/glove/glove.6B.50d_word2id.json'))
word2vec = np.load('../pretrain/glove/glove.6B.50d_mat.npy')
rel2id = json.load(open('../benchmark/semeval/semeval_rel2id.json'))
sentence_encoder = cnn_encoder.SCNNEncoder(token2id=wordi2d,
                                             max_length=100,
                                             word_size=50,
                                             position_size=0,
                                             hidden_size=230,
                                             blank_padding=True,
                                             kernel_size=3,
                                             padding_size=1,
                                             word2vec=word2vec,
                                             dropout=0.5)
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
framework = opennre.framework.SentenceRE(
    train_path='../benchmark/semeval/semantics/train.txt',
    val_path='../benchmark/semeval/semantics/val.txt',
    test_path='../benchmark/semeval/semantics/test.txt',
    model=model,
    ckpt=ckpt,
    batch_size=32,
    max_epoch=100,
    lr=0.1,
    weight_decay=1e-5,
    opt='sgd')
# Train
framework.train_model(metric='micro_f1')
# Test
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)

print('Micro Precision: {}'.format(result['micro_p']))
print('Micro Recall: {}'.format(result['micro_r']))
print('Micro F1: {}'.format(result['micro_f1']))
