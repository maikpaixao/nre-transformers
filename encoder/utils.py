
class Utils:
    def __init__(pos=False):
        self.pos = pos

    def formatr(self, tokens):
        re_tokens = ['[CLS]']
        for token in tokens:
            token = token.lower()
            re_tokens.append(token)
        re_tokens.append('[SEP]')
        return re_tokens
