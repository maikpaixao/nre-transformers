# coding:utf-8
import torch
import numpy as np
import json
sys.path.append("..")
from encoder import cnn_encoder
from model import softmax_nn
from framework import sentence_re

ckpt = 'ckpt/semeval_cnn_softmax.pth.tar'

wordi2d = json.load(open('../benchmark/glove/glove.6B.50d_word2id.json'))
word2vec = np.load('../benchmark/glove/glove.6B.50d_mat.npy')
rel2id = json.load(open('../benchmark/semeval/semeval_rel2id.json'))

sentence_encoder = cnn_encoder.CHUNKCNNEncoder(token2id=wordi2d, max_length=100, word_size=50,
                                            position_size=0, path_entity=50, hidden_size=230, blank_padding=True,
                                            kernel_size=3, padding_size=1, word2vec=word2vec, dropout=0.5)
                                            
model = softmax_nn.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
framework = sentence_re.SentenceRE(train_path='../benchmark/semeval/train.txt',
                                    val_path='../benchmark/semeval/val.txt',
                                    test_path='../benchmark/semeval/test.txt',
                                    model=model, ckpt=ckpt, batch_size=32, max_epoch=100,
                                    lr=0.1, weight_decay=1e-5, opt='sgd')

# Train
framework.train_model(metric='micro_f1')

# Test
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)

f = open("output_cnn_softmax.txt","w+")
f.write('Precision: {}' + result['micro_p'] + '\n')
f.write('Recall: {}' + result['micro_r'] + '\n')
f.write('F1 Score: {}' + result['micro_f1'] + '\n')
f.close()
