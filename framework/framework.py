import json
from sklearn.metrics import classification_report
import torch
from dataloader.dataloader import MyDataset, collate_fn
from torch.utils.data import DataLoader
from models.model import SoftWrapperPrompt
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
# import json

class Framework():
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(self.config.schema, "r", encoding="utf-8") as f:
            self.id2label = json.load(f)[1]

    def train(self):
        train_dataset = MyDataset(self.config, self.config.train_file)
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)

        dev_dataset = MyDataset(self.config, self.config.dev_file)
        dev_dataloader = DataLoader(dev_dataset, batch_size=self.config.eval_batch, collate_fn=collate_fn)

        def loss_func(inputs, targets):
            loss = F.cross_entropy(inputs, targets, reduction="none")
            return loss

        loss_fn = torch.nn.CrossEntropyLoss()
        model = SoftWrapperPrompt(self.config).to(self.device)
        optimizer = torch.optim.AdamW(model.prompt_embedding.parameters(), lr=self.config.learning_rate)
        for p in model.model.parameters():
            p.requires_grad = False

        global_step = 0
        global_loss = 0
        for epoch in range(1, self.config.epochs+1):
            print('{}/{}'.format(epoch, self.config.epochs))
            for data in tqdm(train_dataloader):
                optimizer.zero_grad()
                logits = model(data)
                loss = loss_fn(logits, data["label2id"].to(self.device))
                global_loss += loss.item()

                loss.backward()
                optimizer.step()
            if epoch % 5 ==0:
                dev_result = self.evaluate(model, dev_dataloader)
                # print(dict(dev_result))
                print(dev_result)
            print("loss: {:5.4f}".format(global_loss))
            global_loss = 0

        torch.save(model.state_dict(), self.config.save_model)

    def evaluate(self, model, dataloader):
        print("evulate......")
        model.eval()
        predict = []
        target = []
        with torch.no_grad():
            for data in tqdm(dataloader):
                target.extend(data["label"])
                logits = model(data)
                pred = torch.argmax(logits, dim=-1)
                # print(logits)
                for i in pred.cpu().tolist():
                    predict.append(self.id2label[str(i)])
        model.train()
        return classification_report(np.array(target), np.array(predict))

