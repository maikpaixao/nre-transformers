
import nltk
import pandas as pd
import pickle
import json
from nltk.corpus import wordnet

class SemanticKNWL:
    def __init__(self):
        self.vocabulary = []

    def extract(self, entities):
        ent_dict = []
        for entity in entities:
            #self.vocabulary.append(entity)
            if len(wordnet.synsets(entity)) == 0:
                ent_vector = [entity, entity, entity, entity]
            else:
                syn = wordnet.synsets(entity)[0].hypernyms()
                if len(syn) == 0:
                    ent_vector = [entity, entity, entity, entity]
                else:
                    synonyms_1, synonyms_2 = self.synonyms(entity)
                    ent_vector = [entity, self.unigram(syn[0].name()[:-5]), self.unigram(synonyms_1), self.unigram(synonyms_2)]
            ent_dict.append(ent_vector)
        self.add(ent_dict[0] + ent_dict[1])
        return {'ses1': ent_dict[0], 'ses2': ent_dict[1]}

    def synonyms(self, keyword) :
        synonyms = []
        for synset in wordnet.synsets(keyword):
            for lemma in synset.lemmas():
                synonyms.append(lemma.name())
        if len(synonyms) == 1:
            return str(keyword), str(synonyms[0])
        elif len(synonyms) == 2:
            return str(synonyms[0]), str(synonyms[1])
        else:
            return str(synonyms[1]), str(synonyms[2])

    def get_dir(self, doc, size):
        _index = get_root(doc)
        _dir = []
        for token in doc:
                if token.i == _index:
                    _dir.append('-')
                elif token.i < _index:
                    _dir.append('<')
                elif token.i > _index:
                    _dir.append('>')
        return _dir

    def get_root(self, doc):
        for token in doc:
            if token.dep_ == 'ROOT':
                idroot = token.i
        return idroot

    def format(self, dict):
        text = json.loads(dict)['token']
        text = ' '.join(text)
        entity_1 = json.loads(dict)['h']['name']
        entity_2 = json.loads(dict)['t']['name']
        return text, entity_1, entity_2

    def unigram(self, entity):
        entity = list(entity.split("_"))
        return entity[-1]

    def backoff(self, entity):
        entity = list(entity.split(" "))
        return entity[-1]

    def add(self, lst):
        for token in lst:
            self.vocabulary.append(token)