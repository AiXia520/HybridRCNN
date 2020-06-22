#!/usr/bin/env python
# coding:utf-8
"""
  #laha_xml
  #Label-aware Document Representation via Hybrid Attention for Extreme Multi-Label Text Classification
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from dataset.classification_dataset import ClassificationDataset as cDataset
from model.classification.classifier import Classifier
from model.rnn import RNN
import numpy as np

class TextLAHA(Classifier):
    """Texthybrid_xml
    """
    def __init__(self, dataset, config):
        super(TextLAHA, self).__init__(dataset, config)
        num_labels=config.TextLAHA.num_labels
        num_label_dimension=config.TextLAHA.num_label_dimension
        hidden_dimension = config.TextLAHA.hidden_dimension

        self.embedding_dropout=nn.Dropout(p=0.25, inplace=True)
        self.lstm = nn.LSTM(input_size=config.embedding.dimension, hidden_size=config.TextLAHA.hidden_dimension,
                            num_layers=config.TextLAHA.num_layers,
                            batch_first=True, bidirectional=config.TextLAHA.bidirectional)

        self.label_emb=config.TextLAHA.label_emb
        self.hidden_size = config.TextLAHA.hidden_dimension
        self.batch_size=config.train.batch_size

        # interaction-attn layer
        self.key_layer = torch.nn.Linear(2*hidden_dimension, num_label_dimension)
        self.query_layer = torch.nn.Linear(num_label_dimension, num_label_dimension)

        # self-attn layer
        self.linear_first = torch.nn.Linear(2*hidden_dimension, num_label_dimension)
        self.linear_second = torch.nn.Linear(num_label_dimension, num_labels)

        # weight adaptive layer
        self.linear_weight1 = torch.nn.Linear(2*hidden_dimension, 1)
        self.linear_weight2 = torch.nn.Linear(2*hidden_dimension, 1)

        # prediction layer
        self.linear_final = torch.nn.Linear(2*hidden_dimension, hidden_dimension)
        self.output_layer = torch.nn.Linear(hidden_dimension, 1)
        # self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)

    def init_hidden(self,batch_size):
        if torch.cuda.is_available():
            return (torch.zeros(2,batch_size,self.hidden_size).cuda(),torch.zeros(2,batch_size,self.hidden_size).cuda())
        else:
            return (torch.zeros(2,batch_size,self.hidden_size),torch.zeros(2,batch_size,self.hidden_size))


    def get_parameter_optimizer_dict(self):
        params = list()
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.char_embedding.parameters()})
        params.append({'params': self.embedding_dropout.parameters()})
        params.append({'params': self.lstm.parameters()})
        params.append({'params': self.key_layer.parameters()})
        params.append({'params': self.query_layer.parameters()})
        params.append({'params': self.linear_first.parameters()})
        params.append({'params': self.linear_second.parameters()})
        params.append({'params': self.linear_weight1.parameters()})
        params.append({'params': self.linear_weight2.parameters()})
        params.append({'params': self.linear_final.parameters()})
        params.append({'params': self.output_layer.parameters()})
        return params

    def update_lr(self, optimizer, epoch):
        """
        """
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = self.config.optimizer.learning_rate
        else:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = 0

    def forward(self, batch):
        if self.config.feature.feature_names[0] == "token":
            embedding = self.token_embedding(
                batch[cDataset.DOC_TOKEN].to(self.config.device))
            seq_length = batch[cDataset.DOC_TOKEN_LEN].to(self.config.device)
        else:
            embedding = self.char_embedding(
                batch[cDataset.DOC_CHAR].to(self.config.device))
            seq_length = batch[cDataset.DOC_CHAR_LEN].to(self.config.device)

        # LSTM start
        # output, _ = self.rnn(embedding, seq_length) #[batch, seq, 2*hidden]

        hidden_state = self.init_hidden(seq_length.size(0))
        output, hidden_state = self.lstm(embedding, hidden_state)  # [batch,seq,2*hidden]

        # get attn_key
        attn_key = self.key_layer(output)  # [batch,seq,hidden]
        attn_key = attn_key.transpose(1, 2)  # [batch,hidden,seq]
        # get attn_query
        label_emb = self.label_emb.expand((attn_key.size(0), self.label_emb.size(0), self.label_emb.size(1)))  # [batch,L,label_emb]
        label_emb = self.query_layer(label_emb.cuda())  # [batch,L,label_emb]

        # attention
        similarity = torch.bmm(label_emb, attn_key)  # [batch,L,seq]
        similarity = F.softmax(similarity, dim=2)

        out1 = torch.bmm(similarity, output)  # [batch,L,2*hidden]

        # self-attn output
        self_attn = torch.tanh(self.linear_first(output))  # [batch,seq,d_a]
        self_attn = self.linear_second(self_attn)  # [batch,seq,L]
        self_attn = F.softmax(self_attn, dim=1)
        self_attn = self_attn.transpose(1, 2)  # [batch,L,seq]
        out2 = torch.bmm(self_attn, output)  # [batch,L,2*hidden]

        # normalize
        out1 = F.normalize(out1, p=2, dim=-1)
        out2 = F.normalize(out2, p=2, dim=-1)

        factor1 = torch.sigmoid(self.linear_weight1(out1))
        factor2 = torch.sigmoid(self.linear_weight2(out2))

        factor1 = factor1 / (factor1 + factor2 )
        factor2 = 1 - factor1

        out = factor1 * out1+ factor2 * out2

        out = F.relu(self.linear_final(out), inplace=True)
        # out = torch.sigmoid(self.output_layer(out).squeeze(-1))  # [batch,L]
        out = self.output_layer(out).squeeze(-1)  # [batch,L]
        return out