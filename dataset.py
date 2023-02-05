from torch.utils.data import Dataset
from hazm import WordTokenizer
from random import shuffle


class UnscramblingInFormalDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = WordTokenizer()

    def __getitem__(self, item):
        row = self.data[item]
        words = self.tokenizer.tokenize(row[0])
        shuffle(words)
        return (
            words, row[1]
        )

    def __len__(self):
        return len(self.df)


