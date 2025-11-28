import os
from datetime import datetime

from mltu.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = "Models/DetectionModel/model"
        # self.vocab = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.height = 512
        self.width = 512
        # self.max_text_length = 23
        self.batch_size = 512
        self.learning_rate = 1e-4
        self.train_epochs = 100
        self.train_workers = 20
        chanels = 3