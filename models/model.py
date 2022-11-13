import torch
from transformers import BertForMaskedLM, BertTokenizer
import torch.nn as nn


class PromptEmbedding(nn.Module):
    def __init__(self, config):
        super(PromptEmbedding, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config.prompt_token_num, 768)
        # self.lstm = nn.LSTM(8, 768 // 2, bidirectional=True, batch_first=True)

    def forward(self, input_ids):
        # 把不属于bert vocab 的词（prompt token ids） 取出来编码
        prompt_token_ids = input_ids - self.config.bert_vocab
        emb = self.embedding(prompt_token_ids)
        return emb

class SoftWrapperPrompt(nn.Module):
    def __init__(self, config):
        super(SoftWrapperPrompt, self).__init__()
        self.config = config
        self.prompt_embedding = PromptEmbedding(self.config)
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)
        self.label_list = [self.tokenizer.convert_tokens_to_ids("差"), self.tokenizer.convert_tokens_to_ids("好")]
        self.model = BertForMaskedLM.from_pretrained(self.config.bert_path)
        self.prompt = lambda t: (t >= self.config.bert_vocab)
        self.bert_embedding = self.model.get_input_embeddings()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):
        input_ids = data["input_ids"].to(self.device)
        mask = data["mask"].to(self.device)

        input_ids_ = input_ids.clone()
        # [False, True, True,...False]
        prompt = self.prompt(input_ids)

        input_ids_[prompt] = 0

        bert_embedding = self.bert_embedding(input_ids_)

        prompt_embedding = self.prompt_embedding(input_ids[prompt])

        bert_embedding[prompt] = prompt_embedding
        # print(self.model(bert_embedding, attention_mask=mask))
        lm_logits = self.model(inputs_embeds = bert_embedding, attention_mask=mask).logits
        lm_logits = lm_logits[:, self.config.mask_idx]
        lm_logits = lm_logits[:, self.label_list]

        return lm_logits

