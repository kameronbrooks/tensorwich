# tensorwich

Tensorwich is a python package for saving and loading embedding data for pandas dataframe columns as a compact binary format


# Features

- Save and load embedding data for pandas dataframe columns as a compact binary format that accompanies the saved csv file


# Installation

```bash
pip install tensorwich
```

# Usage

```python
import tensorwich as tw
import pandas as pd

# load a df from a csv file and wrap it in a tensorwich
df = pd.read_csv("examples/example_df.csv")
tdf = tw.tensorwich(df)

# add the embedding to the tensorwich/df by using a custom function that generates embeddings
# in this case, we are using the "get_embeddings" function to generate embeddings from the Description column
# of the df
tdf.add_embedding('short_desc_embedding', get_embeddings(tdf.df['Description'].tolist()))

# save the df and related tensors to disk
tdf.save("examples/example_df_with_embeddings.csv")

# load the related tensors from disk
result = tw.tensorwich.load("examples/example_df_with_embeddings.csv")

```

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
```


