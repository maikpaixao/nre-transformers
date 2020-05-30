# coding:utf-8
import torch
import numpy as np
import json
import sys
sys.path.append("..")
from encoder import bert_encoder
from model import softmax_nn
from framework import sentence_re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--position", help="position embeddings", action="store_true")
parser.add_argument("--path", help="path embeddings", action="store_true")
parser.add_argument("--chunks", help="chunks embeddings", action="store_true")
parser.add_argument("--semantics", help="semantics embeddings", action="store_true")
args = parser.parse_args()

def main():
    ckpt = 'ckpt/semeval_bert_softmax.pth.tar'
    wordi2d = json.load(open('../benchmark/glove/glove.6B.50d_word2id.json'))
    word2vec = np.load('../benchmark/glove/glove.6B.50d_mat.npy')
    rel2id = json.load(open('../benchmark/semeval/semeval_rel2id.json'))

    sentence_encoder = bert_encoder.BERTEncoder(token2id = wordi2d, max_length=80,
                                            pretrain_path='../benchmark/bert-base-uncased', word2vec=word2vec,
                                            e_position = args.position, e_path = args.path,
                                            e_chunks = args.chunks, e_semantics = args.semantics)
                
    model =softmax_nn.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
    framework = sentence_re.SentenceRE(train_path='../benchmark/semeval/train.txt',
                                        val_path='../benchmark/semeval/val.txt',
                                        test_path='../benchmark/semeval/test.txt',
                                        model=model, ckpt=ckpt, batch_size=64,
                                        max_epoch=10, lr=3e-5, opt='adam')

    framework.train_model(metric='micro_f1')
    framework.load_state_dict(torch.load(ckpt)['state_dict'])
    result = framework.eval_model(framework.test_loader)

    f = open("output_bert.txt","w+")
    f.write('Precision: ' + str(result['micro_p']) + '\n')
    f.write('Recall: ' + str(result['micro_r']) + '\n')
    f.write('Score: ' + str(result['micro_f1']) + '\n')
    f.close()

if __name__ == '__main__'():
    main()
    