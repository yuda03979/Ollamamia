from chitchat import *

class Models:

    def __init__(self):
        self.models = {}

    def add(self, model_name):
        self.models[model_name] = Model(model_name=model_name)

    def infer(self, model_name, task, param):
        return self.models[model_name].get_tasks()[task](param)

    def close(self):
        [model.close() for model in self.models.values()]


MODELS = Models()