import os

class path:
    def __init__(self, file):
        self.path = path = os.path.realpath('..') + "\\dataset\\" + file
