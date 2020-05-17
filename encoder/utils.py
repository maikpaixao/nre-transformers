
class Utils:
    def __init__(self, cnn=False):
        self.cnn = cnn
        self.pos = 0

    def formatr(self, tokens):
        if self.cnn == False:
            re_tokens = ['[CLS]']
            for token in tokens:
                token = str(token).lower()
                re_tokens.append(token)
            re_tokens.append('[SEP]')
        else:
            re_tokens = []
            for token in tokens:
                token = str(token).lower()
                re_tokens.append(token)
        return re_tokens

    def save(self, tensors):
        return tensors