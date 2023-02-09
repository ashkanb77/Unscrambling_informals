import pandas as pd
from sklearn.model_selection import train_test_split


def read_dataset(train_informal_path, column_names, split=True, last_5_percent=False):
    df = pd.read_csv(train_informal_path)
    df = df[column_names]

    if last_5_percent:
        df = df.tail(int(len(df) * 0.05))
    else:
        df = df.head(int(len(df) * 0.95))
        df = df.sample(int(len(df) * 0.2))

    df.dropna(inplace=True)

    if split:
        train_df, val_df = train_test_split(df.values, test_size=0.1, random_state=7)
        return train_df, val_df

    return df.values, df.values


def collate_fn(data, tokenizer):
    informal_words, formal = zip(*data)
    formal = list(formal)
    informal_words = list(informal_words)

    tokenized = tokenizer(informal_words, text_target=formal, padding=True, return_tensors='pt')
    return tokenized
