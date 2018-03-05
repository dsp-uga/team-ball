"""Abstract calss for preprocessors"""

class preprocessor:

    def __init__(self):
        self.data= None

    def loadSample(self, path):
        self.data = None

    def preprocess (self):
        return None