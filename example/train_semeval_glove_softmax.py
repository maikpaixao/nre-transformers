# coding:utf-8
import torch
import numpy as np
import json
import opennre
from openlis import encoder, model, framework

ckpt = 'ckpt/semeval_cnn_softmax.pth.tar'
wordi2d = json.load(open('../pretrain/glove/glove.6B.50d_word2id.json'))
word2vec = np.load('../pretrain/glove/glove.6B.50d_mat.npy')
rel2id = json.load(open('../benchmark/semeval/semeval_rel2id.json'))
sentence_encoder = opennre.encoder.GloveEncoder(token2id=wordi2d,
                                             max_length=100,
                                             word_size=50,
                                             position_size=5,
                                             hidden_size=230,
                                             blank_padding=True,
                                             word2vec=word2vec)
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
framework = opennre.framework.SentenceRE(
    train_path='../benchmark/semeval/semeval_train.txt',
    val_path='../benchmark/semeval/semeval_val.txt',
    test_path='../benchmark/semeval/semeval_test.txt',
    model=model,
    ckpt=ckpt,
    batch_size=64,
    max_epoch=10,
    lr=3e-5,
    opt='adam')
# Train
framework.train_model(metric='micro_f1')
# Test
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)
print('Accuracy on test set: {}'.format(result['acc']))
print('Micro Precision: {}'.format(result['micro_p']))
print('Micro Recall: {}'.format(result['micro_r']))
print('Micro F1: {}'.format(result['micro_f1']))
