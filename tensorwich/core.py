import pandas as pd
import numpy as np

class TensorwichWrapper:
    def __init__(self, df:pd.DataFrame):
        self.df = df
        self.embeddings = None
    
    def save_file(self, path:str, *, embedding_column_name=None):
        # TODO : Implement save function
        pass

    def generate_embeddings(self, target_column:str, *, model:any=None, embedding_func:callable=None, column_name:str=None):
        # TODO : Implement generate_embeddings function
        pass

    def load_file(self, path:str, *, embedding_column_name='embedding'):
        # TODO : Implement load function
        pass

    