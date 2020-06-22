#!usr/bin/env python
# coding:utf-8
"""
Tencent is pleased to support the open source community by making NeuralClassifier available.
Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the MIT License (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at
http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for thespecific language governing permissions and limitations under
the License.
"""

import torch

from dataset.classification_dataset import ClassificationDataset as cDataset
from model.classification.classifier import Classifier
from model.rnn import RNN
from model.layers import SumAttention
from util import Type

class DocEmbeddingType(Type):
    """Standard names for doc embedding type.
    """
    AVG = 'AVG'
    ATTENTION = 'Attention'
    LAST_HIDDEN = 'LastHidden'

    @classmethod
    def str(cls):
        return ",".join(
            [cls.AVG, cls.ATTENTION, cls.LAST_HIDDEN])
class TextCRVariant(Classifier):
    def __init__(self, dataset, config):
        super(TextCRVariant, self).__init__(dataset, config)

        self.label_semantic_emb=config.TextCRVariant.label_semantic_emb

        self.doc_embedding_type = config.TextCRVariant.doc_embedding_type
        self.kernel_sizes = config.TextCRVariant.kernel_sizes
        self.convs = torch.nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            self.convs.append(torch.nn.Conv1d(
                384, config.TextCRVariant.num_kernels,
                kernel_size, padding=kernel_size - 1))

        self.top_k = self.config.TextCRVariant.top_k_max_pooling
        hidden_size = len(config.TextCRVariant.kernel_sizes) * \
                      config.TextCRVariant.num_kernels * self.top_k

        self.rnn3=torch.nn.GRU(256,256, num_layers=config.TextCRVariant.num_layers, batch_first=True)
        self.rnn1 = RNN(
            config.embedding.dimension, config.TextCRVariant.hidden_dimension,
            num_layers=config.TextCRVariant.num_layers, batch_first=True,
            bidirectional=config.TextCRVariant.bidirectional,
            rnn_type=config.TextCRVariant.rnn_type)

        self.rnn2= RNN(
            384, config.TextCRVariant.hidden_dimension,
            num_layers=config.TextCRVariant.num_layers, batch_first=True,
            bidirectional=config.TextCRVariant.bidirectional,
            rnn_type=config.TextCRVariant.rnn_type)

        hidden_dimension = config.TextCRVariant.hidden_dimension
        if config.TextCRVariant.bidirectional:
            hidden_dimension *= 2

        self.sum_attention = SumAttention(hidden_dimension,
                                          config.TextCRVariant.attention_dimension,
                                          config.device)

        self.linear = torch.nn.Linear(hidden_size, len(dataset.label_map))
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)

    def get_parameter_optimizer_dict(self):
        params = list()
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.char_embedding.parameters()})
        params.append({'params': self.convs.parameters()})
        params.append({'params': self.rnn1.parameters()})
        params.append({'params': self.rnn2.parameters()})
        params.append({'params': self.linear.parameters()})

        return params

    def update_lr(self, optimizer, epoch):
        """Update lr
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

        label_semantic_emb=self.label_semantic_emb.cuda()

        label_semantic_emb = label_semantic_emb.expand(
            (embedding.size(0), label_semantic_emb.size(0), label_semantic_emb.size(1)))  # [batch,L,256]

        output1, last_hidden = self.rnn3(label_semantic_emb)  # [batch,n,256]

        last_hidden=last_hidden.squeeze(dim=0)

        last_hidden = last_hidden.expand(
            (embedding.size(1), last_hidden.size(0), last_hidden.size(1)))  # [batch,n,256]

        last_hidden=last_hidden.transpose(0,1)

        doc_embedding, _ = self.rnn1(embedding, seq_length) #[batch,n,256]

        input= torch.cat((doc_embedding,last_hidden), 2)  ##[batch,512]

        doc_embedding = input.transpose(1, 2)
        pooled_outputs = []
        for _, conv in enumerate(self.convs):
            convolution = torch.nn.functional.relu(conv(doc_embedding))
            pooled = torch.topk(convolution, self.top_k)[0].view(
                convolution.size(0), -1)
            pooled_outputs.append(pooled)

        doc_embedding = torch.cat(pooled_outputs, 1)

        out=self.dropout(self.linear(doc_embedding))

        return out
