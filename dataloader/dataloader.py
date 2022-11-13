import json

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer
from utils.utils import generate_template, extend_tokenizers
import numpy as np


class MyDataset(Dataset):
    def __init__(self, config, fn):
        self.config = config
        self.df = pd.read_csv(fn)
        self.label = self.df.label.tolist()
        self.sentence = self.df.sentence.tolist()
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)
        self.tokenizer = extend_tokenizers(self.tokenizer, self.config.prompt_token_num)
        with open(self.config.schema, "r", encoding="utf-8") as f:
            self.label2id = json.load(f)[0]

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        label = self.label[idx]
        label2id = self.label2id[label]

        sentence = self.sentence[idx][:self.config.max_length]
        template = generate_template(self.config.template, self.config.prompt_token_num)
        sent = template + sentence
        token = ['[CLS]'] + self.tokenizer.tokenize(sent) + ['[SEP]']
        token_len = len(token)
        convert_tokens_to_ids = self.tokenizer.convert_tokens_to_ids(token)

        input_ids = np.array(convert_tokens_to_ids)
        mask = np.array([1] * token_len)
        return sentence, label, input_ids, mask, label2id, token_len

def collate_fn(batch):
    sentence, label, input_ids, mask, label2id, token_len = zip(*batch)
    cur_batch = len(batch)
    max_token_len = max(token_len)

    batch_input_ids = torch.LongTensor(cur_batch, max_token_len).zero_()
    batch_mask = torch.LongTensor(cur_batch, max_token_len).zero_()

    for i in range(cur_batch):
        batch_input_ids[i, :token_len[i]].copy_(torch.from_numpy(input_ids[i]))
        batch_mask[i, :token_len[i]].copy_(torch.from_numpy(mask[i]))

    return {"input_ids": batch_input_ids,
            "mask": batch_mask,
            "sentence": sentence,
            "label": label,
            "label2id": torch.tensor(label2id, dtype=torch.long)}

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from config.config import Config
    config = Config()

    dataset = MyDataset(config, config.train_file)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for data in dataloader:
        print(data)
