
class Utils:
    def __init__(self, cnn=False):
        self.pos = 0

    def formatr(self, tokens):
        if cnn == False:
            re_tokens = ['[CLS]']
            for token in tokens:
                token = token.lower()
                re_tokens.append(token)
            re_tokens.append('[SEP]')
        else:
            for token in tokens:
                token = token.lower()
                re_tokens.append(token)
        return re_tokens
