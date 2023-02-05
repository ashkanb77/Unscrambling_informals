import pandas as pd
from sklearn.model_selection import train_test_split


def read_dataset(train_informal_path, split=True):
    df = pd.read_csv(train_informal_path)
    df.rename(columns={"formalForm": "FormalForm"}, inplace=True)
    df = df[['inFormalForm', 'FormalForm']]

    df.dropna(inplace=True)

    if split:
        train_df, val_df = train_test_split(df.values, test_size=0.1, random_state=7)
        return train_df, val_df

    return df.values


def collate_fn(data, tokenizer):
    informal_words, formal = zip(*data)
    formal = list(formal)
    informal_words = list(informal_words)

    tokenized = tokenizer(informal_words, text_target=formal, padding=True, return_tensors='pt')
    return tokenized
