import numpy as np
import pickle
import json

class PathsKNWL:
    def __init__(self, path):
        self.path = path
        self.dict_lemma, self.dict_pos, self.dict_dep, self.dict_dir = self.get_vocab()
        self.dict_path = self.extract()
    
    def extract(self):
        paths = self.read()
        dict_path = {}
        c = 0

        for path in paths:
            _path_wv = []
            _cedge_pos = 0

            for _edge in path:
                if isinstance(_edge, list):
                    if _cedge_pos == 0:
                        xs = _edge[0] #satelite lemma X
                        edge = [0, self.dict_pos[_edge[1]], self.dict_dep[_edge[2]], self.dict_dir[_edge[3]]]
                    elif _cedge_pos == (len(path) - 1):
                        ys = _edge[0] #satelite lemma Y
                        edge = [1, self.dict_pos[_edge[1]], self.dict_dep[_edge[2]], self.dict_dir[_edge[3]]]
                    else:
                        edge = [self.dict_lemma[_edge[0]], self.dict_pos[_edge[1]], self.dict_dep[_edge[2]], self.dict_dir[_edge[3]]]

                    _path_wv.append(edge)
                    _cedge_pos = _cedge_pos + 1

            dict_path[c] = {'xs' : xs, 'ys' : ys, 'embed' : _path_wv}
            c = c+1

        return dict_path
    
    def read(self):
        with open('./'+self.path, 'rb') as file:
            try:
                while True:
                    yield pickle.load(file)
            except EOFError:
                pass

    def unique(self, _list):
        unique_list = []
        for x in _list:
                if x not in unique_list:
                        unique_list.append(x)
        return unique_list

    def to_dict(self, lst):
        op = { lst[i] : i for i in range(0, len(lst) ) }
        return op

    def get_vocab(self):
        paths = self.read()
        lemma_dict = []
        pos_dict = []
        dep_dict = []
        dir_dict = []
        for _path in paths:
            for _edge in _path:
                if isinstance(_edge, list):
                    lemma_dict.append(_edge[0])
                    pos_dict.append(_edge[1])
                    dep_dict.append(_edge[2])
                    dir_dict.append(_edge[3])

        return self.to_dict(self.unique(lemma_dict)), self.to_dict(self.unique(pos_dict)), self.to_dict(self.unique(dep_dict)), self.to_dict(self.unique(dir_dict))