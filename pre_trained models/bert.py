import torch
import pickle
from torch import nn
import numpy as np
import pandas as pd
from transformers import *
from sklearn.metrics import roc_curve, auc
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader, random_split

import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel
import torch.nn.functional as F
from bert import *
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:1')


seed=2020
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
from bert_hybridrcnn import *
class InputExample(object):
    def __init__(self, id, text, labels=None):
        self.id = id
        self.text = text
        self.labels = labels

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

def get_train_examples(train_file):
    train_df = pd.read_csv(train_file)
    ids = train_df['ids'].values
    text = train_df['doc_token'].values.astype(str)
    labels = train_df[train_df.columns[2:]].values
    examples = []
    for i in range(len(train_df)):
        examples.append(InputExample(ids[i], text[i], labels=labels[i]))
    return examples


def get_features_from_examples(examples, max_seq_len, tokenizer):
    features = []
    for i,example in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)
        if len(tokens) > max_seq_len - 2:
            tokens = tokens[:(max_seq_len - 2)]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(tokens)
        padding = [0] * (max_seq_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len
        label_ids = [float(label) for label in example.labels]
        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_ids=label_ids))
    return features



def get_dataset_from_features(features):
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.float)
    dataset = TensorDataset(input_ids,
                            input_mask,
                            segment_ids,
                            label_ids)
    return dataset





class BertForMultiLable(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMultiLable, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, head_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                            head_mask=head_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


    if __name__ == '__main__':

        num_labels = 3714
        pretrained_weights = 'bert-base-cased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        model = BertForMultiLable.from_pretrained(pretrained_weights, num_labels=num_labels)
        model.to("cuda:1")

        seq_len = 256
        train_file = 'eurlex4k_train.csv'

        train_examples = get_train_examples(train_file)
        train_features = get_features_from_examples(train_examples, seq_len, tokenizer)
        train_dataset = get_dataset_from_features(train_features)

        val_file = "eurlex4k_test.csv"

        val_examples = get_train_examples(val_file)
        val_features = get_features_from_examples(val_examples, seq_len, tokenizer)
        val_dataset = get_dataset_from_features(val_features)

        batch = 16
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch)
        val_sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch)



        epochs = 100
]
        from tqdm import tqdm
        from sklearn.metrics import f1_score
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=0.01)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

        for i in range(epochs):
            print('-----------EPOCH #{}-----------'.format(i + 1))
            print('training...')
            train_loss = 0
            model.train()
            for step, batch in enumerate(tqdm(train_dataloader)):
                batch = tuple(t.to("cuda:1") for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                pred = model(input_ids, segment_ids, input_mask)
                optimizer.zero_grad()
                loss = criterion(pred, label_ids) / pred.size(0)
                loss.backward()
                optimizer.step()

                train_loss += float(loss)

            batch_num = step + 1
            train_loss /= batch_num

            print("epoch %2d 训练结束 : avg_loss = %.4f" % (i+1, train_loss))
            y_true = []
            y_pred = []

            model.eval()
            print('evaluating...')
            for step, batch in enumerate(tqdm(val_dataloader)):
                batch = tuple(t.to("cuda:1") for t in batch)
                val_input_ids, val_input_mask, val_segment_ids, val_label_ids = batch
                with torch.no_grad():
                    logits = model(val_input_ids, val_segment_ids, val_input_mask)
                    logits = logits.sigmoid()

                y_true.append(val_label_ids)
                y_pred.append(logits)

            y_true = torch.cat(y_true, dim=0).float().cpu().detach().numpy()
            y_pred = torch.cat(y_pred, dim=0).float().cpu().detach().numpy()

            y_pred = (y_pred > 0.5).astype(int)
            f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
            print("F1-micro", f1_micro)
            f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
            print("F1-macro", f1_macro)

