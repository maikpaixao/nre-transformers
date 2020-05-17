
import json
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from knowledge.semantic_knowledge import SemanticKNWL
from knowledge.paths_knowledge import PathsKNWL
from knowledge.chunk_knowledge import ChunkKNWL
from transformers.cnn_transformer import CNNTransformer

import sys
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--prefix', help='prefix file')
args = ap.parse_args()

def main():
    pf = open('data/'+args.prefix+'.txt', 'r+')
    _ofile = open('../benchmark/semeval/'+args.prefix+'.txt', 'w+')

    transformer = CNNTransformer()
    semantic_builder = SemanticKNWL()
    chunks_builder = ChunkKNWL()

    path_builder = PathsKNWL('paths_text/path_'+args.prefix)

    '''
    data_x = transformer.dict_to_dataframe(path_builder.dict_path)
    data_y = get_relations(pf.readlines())['relations']
    label_encoder = LabelEncoder()
    data_y = label_encoder.fit_transform(data_y)
    data_y = to_categorical(data_y)
    transformer.fit(data_x, data_y)
    '''
    count = 0

    for dict in pf.readlines():
        dict_semantics = {}
        text, h, t, entity_1, entity_2, relation = format(dict)
        dict_semantics['token'] = text
        dict_semantics['h'] = h
        dict_semantics['t'] = t
        dict_semantics['path'] = path_builder.dict_path[count]
        dict_semantics['chunks'] = chunks_builder.extract(' '.join(text))
        dict_semantics['semantics'] = semantic_builder.extract([entity_1, entity_2])
        dict_semantics['relation'] = relation

        _ofile.write(json.dumps(dict_semantics))
        _ofile.write('\n')
        count = count + 1

    print('Finished.')

def format(dict):
    text = json.loads(dict)['token']
    h = json.loads(dict)['h']
    t = json.loads(dict)['t']
    entity_1 = json.loads(dict)['h']['name']
    entity_2 = json.loads(dict)['t']['name']
    relation = json.loads(dict)['relation']
    return text, h, t, entity_1, entity_2, relation

def get_relations(dicts):
    relations = pd.DataFrame(columns=['relations'])
    idx = 0
    for dict in dicts:
        _, _, _, _, _, relation = format(dict)
        relations.loc[idx] = [relation]
        idx = idx + 1
    return relations

if __name__ == '__main__':
    main()
