import torch
import pickle
from torch import nn
import numpy as np
import pandas as pd
from transformers import *
from sklearn.metrics import roc_curve, auc
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader, random_split
import torch.nn.functional as F
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device('cuda:3')

seed=2020
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
from xlnet_hybridrcnn import *
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
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    cls_token_segment_id = 1
    pad_token_segment_id = 0

    for i, example in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)
        if len(tokens) > max_seq_len - 2:
            tokens = tokens[:(max_seq_len - 2)]

        tokens = tokens + [sep_token]
        segment_ids = [0] * len(tokens)

        tokens += [cls_token]
        segment_ids += [cls_token_segment_id]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        input_len = len(input_ids)
        padding_len = max_seq_len - len(input_ids)

        # pad on the left for xlnet
        input_ids = ([pad_token] * padding_len) + input_ids
        input_mask = ([0] * padding_len) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_len) + segment_ids

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





# self attention
class LabelSumAttention(torch.nn.Module):

    def __init__(self, input_dimension, attention_dimension, label_dimension, device, dropout=0):
        super(LabelSumAttention, self).__init__()
        self.attention_matrix = \
            torch.nn.Linear(input_dimension, attention_dimension).to(device)
        self.attention_vector = torch.nn.Linear(attention_dimension, label_dimension, bias=False).to(device)
        torch.nn.init.xavier_uniform_(self.attention_matrix.weight)
        torch.nn.init.xavier_uniform_(self.attention_vector.weight)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, inputs):
        if inputs.size(1) == 1:
            return self.dropout(inputs.squeeze())
        u = torch.tanh(self.attention_matrix(inputs))
        v = self.attention_vector(u)
        alpha = torch.nn.functional.softmax(v, 1)
        return self.dropout(torch.matmul(alpha.transpose(1, 2), inputs))


# HybridRCNN
class HybridRCNN(nn.Module):
    def __init__(self, embed_num, embed_dim, label_emb, num_heads, dropout=0.1, num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.label_emb = label_emb
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout, inplace=True)

        # 定义CNN
        self.cnn_q_1 = nn.Conv1d(in_channels=self.embed_dim,
                                 out_channels=128, kernel_size=3)
        self.cnn_k_1 = nn.Conv1d(in_channels=self.embed_dim,
                                 out_channels=128, kernel_size=3)
        self.cnn_v_1 = nn.Conv1d(in_channels=self.embed_dim,
                                 out_channels=128, kernel_size=3)

        self.softmax = nn.Softmax(dim=2)

        self.sum_attention2 = LabelSumAttention(128, 16, self.num_labels, "cuda")

        # 定义rnn,使用GRU模式
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=64, num_layers=1, batch_first=True,
                            bidirectional=True)

        # 定义 self-attention
        self.sum_attention = LabelSumAttention(128, 16, self.num_labels, "cuda")

        # weight adaptive layer
        self.linear1 = torch.nn.Linear(256, 128)  # [256, 128]
        self.linear2 = torch.nn.Linear(256, 128)  # [256, 128]
        self.linear_weight1 = torch.nn.Linear(128, 1)  # [128, 1]
        self.linear_weight2 = torch.nn.Linear(128, 1)  # [128, 1]

        # prediction layer
        self.linear_final = torch.nn.Linear(256, 128)  # [256, 128]
        self.output_layer = torch.nn.Linear(128, 1)  # [128, 1]

    #         self.embed = nn.Embedding(self.embed_num, self.embed_dim)

    #         self.dropout = nn.Dropout(self.dropout)
    #         self.classifier = nn.Linear(len(self.kernel_sizes)*self.kernel_num, self.num_labels)

    def init_hidden(self, batch_size):
        if torch.cuda.is_available():
            return (torch.zeros(2, batch_size, 128).cuda(), torch.zeros(2, batch_size, 128).cuda())
        else:
            return (torch.zeros(2, batch_size, 128), torch.zeros(2, batch_size, 128))

        # [batch_size, seq_length, embedding_size]

    def forward(self, inputs, labels=None):

        # RNN start
        embedding = self.dropout(inputs)  # [batch_size, seq_length, embedding_size]

        hidden_state = self.init_hidden(embedding.size(0))
        output, hidden_state = self.lstm(embedding)  # [batch,seq,128]

        # self-attention
        rnn_out1 = self.sum_attention(output)  # [batch,L,128]
        rnn_out1 = F.normalize(rnn_out1, p=2, dim=-1)
        # interaction-attention label_embedding_dimension:128
        key = output.permute(0, 2, 1)
        label_emb = self.label_emb.expand(
            (key.size(0), self.label_emb.size(0), self.label_emb.size(1)))  # [batch,L,label_emb]
        s = torch.bmm(label_emb.cuda('cuda:3'), key)  # [batch,L,seq]
        s = F.softmax(s, dim=2)
        rnn_out2 = torch.bmm(s, output)  # [batch,L,128]
        rnn_out2 = F.normalize(rnn_out2, p=2, dim=-1)
        rnn_out = torch.cat((rnn_out1, rnn_out2), dim=2)  # [batch,L,256]

        # CNN start
        cnn_embedding = embedding.permute(0, 2, 1)  # [batch,embeedim_dim, max_seq ]
        # multi-head self-attention
        q_1 = (F.elu(self.cnn_q_1(cnn_embedding)))  # q = [batch_Size, emd_dim, seq_len]
        k_1 = (F.elu(self.cnn_k_1(cnn_embedding)))  # k = [batch_Size, emd_dim, seq_len]
        v_1 = (F.elu(self.cnn_v_1(cnn_embedding)))  # v = [batch_Size, emd_dim, seq_len]
        q = torch.chunk(q_1, self.num_heads, 1)
        k = torch.chunk(k_1, self.num_heads, 1)
        v = torch.chunk(v_1, self.num_heads, 1)

        self_heads = []
        for i in range(len(q)):
            a = torch.bmm(q[i].permute(0, 2, 1), k[i])  # a = [batch, seq_len, seq_len]
            a = self.softmax(a / q[i].size(1) ** 0.5)  # a = [batch, seq_len, seq_len]
            h = torch.bmm(a, v[i].permute(0, 2, 1))  # output= [batch, seq_len, emd_dim]
            self_heads.append(h)
        cnn_out1 = torch.cat(tuple(self_heads), dim=-1)  # output= [batch, seq_len, emd_dim]

        cnn_out1 = self.sum_attention2(cnn_out1)  # [batch,L,128]
        cnn_out1 = F.normalize(cnn_out1, p=2, dim=-1)

        # label interaction attention
        b = torch.bmm(label_emb.cuda('cuda:3'), k_1)  # [batch,L,seq]
        b = F.softmax(b, dim=2)
        cnn_out2 = torch.bmm(b, v_1.permute(0, 2, 1))  # [batch,L,128]
        cnn_out2 = F.normalize(cnn_out2, p=2, dim=-1)
        cnn_out = torch.cat((cnn_out1, cnn_out2), dim=2)  # [batch,L,256]

        # weight ensemble
        factor1 = torch.sigmoid(self.linear_weight1(torch.tanh(self.linear1(rnn_out))))
        factor2 = torch.sigmoid(self.linear_weight2(torch.tanh(self.linear2(cnn_out))))

        factor1 = factor1 / (factor1 + factor2)
        factor2 = 1 - factor1

        out = factor1 * rnn_out + factor2 * cnn_out  # [batch,L,256]
        out = F.relu(self.linear_final(out), inplace=True)
        logits = torch.sigmoid(self.output_layer(out).squeeze(-1))  # [batch,L]

        return logits

    if __name__ == '__main__':


        pretrained_weights = 'xlnet-base-cased'
        tokenizer = XLNetTokenizer.from_pretrained(pretrained_weights)
        basemodel = XLNetModel.from_pretrained(pretrained_weights)
        basemodel.to('cuda:3')

        seq_len = 256
        train_file = 'eurlex4k_train.csv'

        train_examples = get_train_examples(train_file)
        train_features = get_features_from_examples(train_examples, seq_len, tokenizer)
        train_dataset = get_dataset_from_features(train_features)

        val_file="eurlex4k_test.csv"

        val_examples = get_train_examples(val_file)
        val_features = get_features_from_examples(val_examples, seq_len, tokenizer)
        val_dataset = get_dataset_from_features(val_features)

        batch = 8
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch)
        val_sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch)

        embed_num = seq_len
        embed_dim = basemodel.config.hidden_size
        dropout = basemodel.config.dropout

        label_emb = np.zeros((3714, 128))
        with open('eurlex4k_edge_128.emb', 'r') as f:
            for index, i in enumerate(f.readlines()):
                if index == 0:
                    continue
                i = i.rstrip('\n')
                n = i.split(' ')[0]
                content = i.split(' ')[1:]

                label_emb[int(n)] = [float(value) for value in content]

        label_emb = torch.from_numpy(label_emb).float()

        num_labels = 3714
        num_heads = 5
        model = HybridRCNN(embed_num, embed_dim, label_emb, num_heads, dropout=dropout, num_labels=num_labels)
        model.to('cuda:3')

        # lr = 0.008
        epochs = 100
        from tqdm import tqdm
        from sklearn.metrics import f1_score

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, betas=(0.9, 0.999),eps=1e-08, weight_decay=4e-5)
        criterion = torch.nn.BCELoss(reduction='sum')

        for i in range(epochs):
            print('-----------EPOCH #{}-----------'.format(i + 1))
            print('training...')
            train_loss = 0
            model.train()
            for step, batch in enumerate(tqdm(train_dataloader)):
                batch = tuple(t.to('cuda:3') for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                #         with torch.no_grad():
                with torch.no_grad():
                    inputs= basemodel(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)


                pred = model(inputs[0], label_ids)
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
                batch = tuple(t.to('cuda:3') for t in batch)
                val_input_ids, val_input_mask, val_segment_ids, val_label_ids = batch
                with torch.no_grad():
                    val_inputs = basemodel(val_input_ids, token_type_ids=val_segment_ids, attention_mask=val_input_mask)
                    logits = model(val_inputs[0])
                y_true.append(val_label_ids)
                y_pred.append(logits)

            y_true = torch.cat(y_true, dim=0).float().cpu().detach().numpy()
            y_pred = torch.cat(y_pred, dim=0).float().cpu().detach().numpy()

            y_pred = (y_pred > 0.5).astype(int)
            f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
            print("F1-micro", f1_micro)
            f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
            print("F1-macro", f1_macro)


















