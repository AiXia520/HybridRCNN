#!/usr/bin/env python
# coding:utf-8
"""
Learning semantic spatio-temporal document representation for extremely multi-label text classification
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from dataset.classification_dataset import ClassificationDataset as cDataset
from model.classification.classifier import Classifier
from model.rnn import RNN
from model.layers import LabelSumAttention,SumAttention
import numpy as np
import math

class hybrid_rnn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.label_emb=config.HybridRCNN.label_emb

        # 定义rnn,使用GRU模式
        self.rnn = RNN(
            config.embedding.dimension, config.HybridRCNN.hidden_dimension,
            num_layers=config.HybridRCNN.num_layers, batch_first=True,
            bidirectional=config.HybridRCNN.bidirectional,
            rnn_type=config.HybridRCNN.rnn_type)

        hidden_dimension = config.HybridRCNN.hidden_dimension
        if config.HybridRCNN.bidirectional:
            hidden_dimension *= 2

        # 定义 self-attention
        self.sum_attention = LabelSumAttention(hidden_dimension,
                                          config.HybridRCNN.attention_dimension,
                                          config.HybridRCNN.num_labels,
                                          config.device)
    def forward(self,embedding,seq_length):
        # rnn start

        output, last_hidden = self.rnn(embedding, seq_length)  # [batch, seq, 128]

        # self-attention
        rnn_out1 = self.sum_attention(output)  # [batch,L,128]
        rnn_out1 = F.normalize(rnn_out1, p=2, dim=-1)
        # interaction-attention label_embedding_dimension:128
        key = output.permute(0, 2, 1)
        label_emb = self.label_emb.expand(
            (key.size(0), self.label_emb.size(0), self.label_emb.size(1)))  # [batch,L,label_emb]
        s = torch.bmm(label_emb.cuda(), key)  # [batch,L,seq]
        s = F.softmax(s, dim=2)
        rnn_out2 = torch.bmm(s, output)  # [batch,L,128]
        rnn_out2 = F.normalize(rnn_out2, p=2, dim=-1)
        rnn_out = torch.cat((rnn_out1, rnn_out2), dim=2)  # [batch,L,256]
        return rnn_out

class hybrid_cnn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.label_emb = config.HybridRCNN.label_emb
        self.num_heads = config.HybridRCNN.num_heads
        # 定义cnn
        self.kernel_sizes = config.HybridRCNN.kernel_sizes

        self.cnn_q_1 = nn.Conv1d(in_channels=config.embedding.dimension,
                                 out_channels=config.HybridRCNN.num_kernels, kernel_size=3)
        self.cnn_k_1 = nn.Conv1d(in_channels=config.embedding.dimension,
                                 out_channels=config.HybridRCNN.num_kernels, kernel_size=3)
        self.cnn_v_1 = nn.Conv1d(in_channels=config.embedding.dimension,
                                 out_channels=config.HybridRCNN.num_kernels, kernel_size=3)

        self.softmax = nn.Softmax(dim=2)

        num_label_dimension = config.HybridRCNN.num_label_dimension
        num_kernels = config.HybridRCNN.num_kernels

        self.sum_attention2 = LabelSumAttention(num_kernels,
                                                config.HybridRCNN.attention_dimension,
                                                config.HybridRCNN.num_labels,
                                                config.device)
    def forward(self,embedding):
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

        label_emb = self.label_emb.expand(
            (k_1.size(0), self.label_emb.size(0), self.label_emb.size(1)))  # [batch,L,label_emb]
        # label interaction attention
        b = torch.bmm(label_emb.cuda(), k_1)  # [batch,L,seq]
        b = F.softmax(b, dim=2)
        cnn_out2 = torch.bmm(b, v_1.permute(0, 2, 1))  # [batch,L,128]
        cnn_out2 = F.normalize(cnn_out2, p=2, dim=-1)
        cnn_out = torch.cat((cnn_out1, cnn_out2), dim=2)  # [batch,L,256]
        return cnn_out


class HybridRCNN(Classifier):
    """HybridRCNN
       Learning semantic spatio-temporal document representation for extremely multi-label text classification
    """
    def __init__(self, dataset, config):
        super(HybridRCNN, self).__init__(dataset, config)
        # 定义label embedding dimension
        self.embedding_dropout=nn.Dropout(p=0.25, inplace=True)
        hidden_dimension = config.HybridRCNN.hidden_dimension
        if config.HybridRCNN.bidirectional:
            hidden_dimension *= 2

        self.hybrid_rnn = hybrid_rnn(config).to('cuda:0')
        self.hybrid_cnn = hybrid_cnn(config).to('cuda:0')

        # weight adaptive layer
        self.linear1 = torch.nn.Linear(2*hidden_dimension, hidden_dimension) #[256, 128]
        self.linear2 = torch.nn.Linear(2*hidden_dimension, hidden_dimension) #[256, 128]
        self.linear_weight1 = torch.nn.Linear(hidden_dimension, 1) #[128, 1]
        self.linear_weight2 = torch.nn.Linear(hidden_dimension, 1) #[128, 1]

        # prediction layer
        self.linear_final = torch.nn.Linear(2*hidden_dimension, hidden_dimension) #[256, 128]
        self.output_layer = torch.nn.Linear(hidden_dimension, 1) #[128, 1]
        # self.linear_final = torch.nn.Linear(512, 256) #[512, 256]
        # self.output_layer = torch.nn.Linear(256, 1) #[256, 1]

    def get_parameter_optimizer_dict(self):
        params = list()
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.char_embedding.parameters()})
        params.append({'params': self.embedding_dropout.parameters()})
        params.append({'params': self.hybrid_cnn.parameters()})
        params.append({'params': self.hybrid_rnn.parameters()})
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

        # embedding:[batch,max_seq,embeddim_dim]

        embedding = self.embedding_dropout(embedding)
        embedding2 = self.embedding_dropout(embedding)
        rnn_out = self.hybrid_rnn(embedding.to('cuda:0'), seq_length.to('cuda:0'))
        cnn_out = self.hybrid_cnn(embedding2.to('cuda:0'))

        # weight ensemble
        factor1 = torch.sigmoid(self.linear_weight1(torch.tanh(self.linear1(rnn_out.to('cuda:0')))))
        factor2 = torch.sigmoid(self.linear_weight2(torch.tanh(self.linear2(cnn_out.to('cuda:0')))))

        factor1 = factor1 / (factor1 + factor2)
        factor2 = 1 - factor1

        out = factor1 * rnn_out+ factor2 * cnn_out # [batch,L,256]
        out = F.relu(self.linear_final(out), inplace=True)
        out = self.output_layer(out).squeeze(-1)  # [batch,L]

        return out