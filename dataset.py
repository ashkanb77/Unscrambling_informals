from torch.utils.data import Dataset
from hazm import WordTokenizer
from random import shuffle, randint


class UnscramblingInFormalDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = WordTokenizer()

    def __getitem__(self, item):
        row = self.data[item]
        words = self.tokenizer.tokenize(row[randint(0, 1)])
        shuffle(words)
        return (
            ' '.join(words), row[1]
        )

    def __len__(self):
        return len(self.data)


