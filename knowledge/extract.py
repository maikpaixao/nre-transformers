
import spacy
import networkx as nx
import stanfordnlp
import pickle
import json
import sys
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--prefix', help='prefix file')
args = ap.parse_args()

nlp = stanfordnlp.Pipeline()
snlp = spacy.load("en_core_web_sm")

fp = open('paths/'+args.prefix, 'wb')
pf = open('data/'+args.prefix+'.txt', 'r+')

def main():
    sdp = []
    for dict in pf.readlines():
        text, entity_1, entity_2 = format(dict)
        doc = nlp(text)
        sdoc = snlp(text)

        edges = []
        for token in doc.sentences[0].dependencies:
            if token[0].text.lower() != 'root':
                edges.append((token[0].text.lower(), token[2].text))

        idx = 0
        deplist = []
        _dir = get_dir(sdoc, len(sdoc))

        for token in sdoc:
            if not token.is_punct | token.is_space:
                deplist.append([token.text, token.lemma_, token.pos_,
                                    token.dep_, _dir[idx]])
            idx = idx + 1

        graph = nx.Graph(edges)
        sdp = nx.shortest_path(graph, source = unigram(str(entity_1).lower()),
                                        target = unigram(str(entity_2).lower()))

        idx = 0
        for w in sdp:
            for ele in deplist:
                if w == ele[0]:
                    sdp[idx] = ele[1:]
            idx = idx + 1
        pickle.dump(sdp, fp)

def get_dir(doc, size):
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

def get_root(doc):
    for token in doc:
        if token.dep_ == 'ROOT':
            idroot = token.i
    return idroot

def format(dict):
    text = json.loads(dict)['token']
    text = ' '.join(text)
    entity_1 = json.loads(dict)['h']['name']
    entity_2 = json.loads(dict)['t']['name']
    return text, entity_1, entity_2

def unigram(entity):
    entity = list(entity.split(" "))
    return entity[-1]

if __name__ == '__main__':
    main()
    