
class Utils:
    def __init__(self):
        self.pos = 0

    def formatr(self, tokens):
        re_tokens = ['[CLS]']
        for token in tokens:
            token = token.lower()
            re_tokens.append(token)
        re_tokens.append('[SEP]')
        return re_tokens
