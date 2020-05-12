import numpy as np
import pickle
import json
import spacy

class ChunkKNWL:
    def __init__(self):
        self.doc_chunks = []
        self.nlp = spacy.load("en_core_web_sm")
    
    def extract(self, text):
        doc = self.nlp(text)
        for chunk in doc.noun_chunks:
            self.doc_chunks.append(chunk.text)
        return self.doc_chunks

