from __future__ import annotations
from typing import List, Tuple, Dict, Any, Union
from io import BufferedReader
import pandas as pd
import numpy as np
import os
import io
from glob import glob



# Binary Format
"""
version 1

| version [4 bytes]               | 
| platform [("pandas\0") 6 bytes] |
| indexType [4 bytes]             |
| tensorDType [4 bytes]           |
| columnName [n bytes]            |
| length [8 bytes]                |
| index byte count [8 bytes]      |
| index data [n bytes]            |
| tensor byte count [8 bytes]     |
| tensor data [n bytes]           |
"""

F16 = 1     # float 16
F32 = 2     # float 32
F64 = 3     # float 64
I8 = 4      # int 8
I16 = 5     # int 16
I32 = 6     # int 32
I64 = 7     # int 64
STR = 8     # string
TUP = 9     # tuple

NUMPY_DTYPE_TABLE = {
    np.float16: F16,
    np.float32: F32,
    np.float64: F64,
    np.int8: I8,
    np.int16: I16,
    np.int32: I32,
    np.int64: I64
}

NUMPY_DTYPE_TABLE_INV = {
    F16: np.float16,
    F32: np.float32,
    F64: np.float64,
    I8: np.int8,
    I16: np.int16,
    I32: np.int32,
    I64: np.int64
}

PANDAS_DTYPE_TABLE = {
    'float16': F16,
    'float32': F32,
    'float64': F64,
    'int8': I8,
    'int16': I16,
    'int32': I32,
    'int64': I64,
    'string': STR,
    'tuple': TUP
}


def _get_column_dtype(df:pd.DataFrame, column_name:str) -> int:
    """
    Get the dtype of a column in a pandas dataframe
    
    Parameters
    ----------
    df : pd.DataFrame
        The pandas dataframe
    column_name : str
        The name of the column to get the dtype of
    """
    return PANDAS_DTYPE_TABLE.get(df[column_name].dtype.name, None)

def _get_index_dtype(df:pd.DataFrame) -> int:
    """
    Get the dtype of the index of a pandas dataframe

    Parameters
    ----------
    df : pd.DataFrame
        The pandas dataframe
    """
    return PANDAS_DTYPE_TABLE.get(df.index.dtype.name, None)

def _get_tensor_dtype(tensor:np.ndarray):
    """
    Get the dtype of a numpy array

    Parameters
    ----------
    tensor : np.ndarray
        The numpy array
    """
    return PANDAS_DTYPE_TABLE.get("float32", None)

def _read_null_terminated_string(f:BufferedReader, encoding='utf-8'):
    """
    Read a null terminated string from a file

    Parameters
    ----------
    f : file
        The file to read from
    """
    if f == None:
        raise ValueError("File is None")
    
    s = b''
    while True:
        if f.eof():
            raise EOFError
        c = f.read(1)
        if c == b'\0':
            break
        s += c
    return s.decode(encoding=encoding)

def _save_embeddings_to_file(path:str, df:pd.DataFrame, embedding_column_name:str='embedding'):
    """
    Save the embeddings to a file

    Parameters
    ----------
    path : str
        The path to save the embeddings to
    df : pd.DataFrame
        The pandas dataframe
    embedding_column_name : str
        The name of the column containing the embeddings
    """
    with open(path, 'wb') as f:
        # Write the version
        f.write(b'\x01\x00\x00\x00')

        # Write the platform
        f.write(b'pandas\0')

        index_type = _get_index_dtype(df)

        # Write the index dtype
        f.write(index_type.to_bytes(4, 'little'))

        tensor_type = _get_tensor_dtype(df[embedding_column_name].iloc[0])

        # Write the tensor dtype
        f.write(tensor_type.to_bytes(4, 'little'))

        # Write the column name
        f.write(embedding_column_name.encode()+b'\0')

        # Write the length
        f.write(len(df).to_bytes(8, 'little'))

        byte_buffer = io.BytesIO()
        np.save(byte_buffer, df.index.to_numpy())

        # Write the index data
        index_data = byte_buffer.getvalue()
        f.write(len(index_data).to_bytes(8, 'little'))
        f.write(index_data)


        # Write the tensor data
        byte_buffer = io.BytesIO()
        np.save(byte_buffer, df[embedding_column_name].to_numpy())
        tensor_data = byte_buffer.getvalue()

        f.write(len(tensor_data).to_bytes(8, 'little'))
        f.write(tensor_data)

def _load_embeddings_from_file(path:str):
    """
    Load the embeddings from a file

    Parameters
    ----------
    path : str
        The path to load the embeddings from
    """
    with open(path, 'rb') as f:
        # Read the version
        version = int.from_bytes(f.read(4), 'little')

        # Read the platform
        platform = _read_null_terminated_string(f)

        # Read the index dtype
        index_dtype =  f.read(4)

        # Read the tensor dtype
        tensor_dtype = NUMPY_DTYPE_TABLE_INV.get(f.read(4), np.float32)

        # Read the column name
        column_name = _read_null_terminated_string(f)

        # Read the length
        length = int.from_bytes(f.read(8), 'little')

        index_buffer_size = int.from_bytes(f.read(8), 'little')
        index_buffer = f.read(index_buffer_size)

        tensor_buffer_size = int.from_bytes(f.read(8), 'little')
        tensor_buffer = f.read(tensor_buffer_size)

        index = np.load(io.BytesIO(index_buffer))
        tensor = np.load(io.BytesIO(tensor_buffer))
        
        return pd.DataFrame(tensor, index=index, columns=[column_name])


class tensorwich:
    def __init__(self, df:pd.DataFrame):
        self.df:pd.DataFrame = df
        self.embedding_columns = []
    
    def save(self, path:str, csv_args:Dict[str, Any]=None, **kwargs):
        if csv_args is None:
            csv_args = {}
        
        # Get the base path
        base_path = path
        if base_path.endswith('.csv'):
            base_path = base_path[:-4]

        output_columns = [col for col in self.df.columns if col not in self.embedding_columns] 

        # save the CSV
        self.df[output_columns].to_csv(base_path+".csv", **csv_args)

        # save the embeddings
        for i, column_name in enumerate(self.embedding_columns):
            _save_embeddings_to_file(base_path+f".{column_name}.emb", self.df[[column_name]], embedding_column_name=column_name)

    
    def add_embedding(self, column_name:str, embedding:np.ndarray):

        if self.df is None:
            raise ValueError("This object does not have a dataframe yet")
        if column_name in self.df.columns:
            raise ValueError(f"Column '{column_name}' already exists in dataframe")
        if len(embedding) != self.df.shape[0]:
            raise ValueError(f"Embedding length ({len(embedding)}) does not match dataframe length ({self.df.shape[0]})")
        
        self.df[column_name] = embedding
        self.embedding_columns.append(column_name)


    def generate_embeddings(self, target_column:str, *, model:any=None, embedding_func:callable=None, column_name:str=None):
        # TODO : Implement generate_embeddings function
        pass

    def load(self, path:str, *, embedding_column_name:str='embedding'):
        # Get the base path
        base_path = path
        if base_path.endswith('.csv'):
            base_path = base_path[:-4]
        
        glob_path = base_path+"*"
        files = glob(glob_path)
        if len(files) == 0:
            raise FileNotFoundError(f"No files found at path {glob_path}")
        
        # Load the CSV
        self.df = pd.read_csv(base_path+".csv")

        embeddings = []
        # Load the embeddings
        for file in files:
            if file.endswith('.csv'):
                continue

            embedding = _load_embeddings_from_file(file)
            embeddings.append(embedding)
            
        


    

    