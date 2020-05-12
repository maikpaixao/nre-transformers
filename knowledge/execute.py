
import json
import numpy as np
import pickle
from knowledge.semantic_knowledge import SemanticKNWL
from knowledge.paths_knowledge import PathsKNWL
from knowledge.chunk_knowledge import ChunkKNWL
import sys
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--prefix', help='prefix file')
args = ap.parse_args()

def main():
    pf = open('data/'+args.prefix+'.txt', 'r+')
    _ofile = open('output/'+args.prefix+'.txt', 'w+')

    semantic_builder = SemanticKNWL()
    chunks_builder = ChunkKNWL()
    path_builder = PathsKNWL('paths_text/path_'+args.prefix)

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

if __name__ == '__main__':
    main()    
