
class Config():
    def __init__(self):
        self.batch_size = 8
        self.learning_rate = 1e-3
        self.epochs = 40
        self.eval_batch = 32
        self.schema = "./dataset/schema.json"
        self.train_file = "./dataset/train.csv"
        self.dev_file = "./dataset/train.csv"
        self.mask_idx = 3
        self.prompt_token_num = 16
        self.max_length = 256
        self.template = "<template_1><template_2>[MASK]"
        self.bert_path = "bert-base-chinese"
        self.save_model = "checkpoint/p_tuning_model_mobu.pt"
        self.bert_vocab = 21128