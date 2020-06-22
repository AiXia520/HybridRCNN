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
class TextGRA(Classifier):
    def __init__(self, dataset, config):
        super(TextGRA, self).__init__(dataset, config)
        self.doc_embedding_type = config.TextGRA.doc_embedding_type
        self.kernel_sizes = config.TextGRA.kernel_sizes
        self.convs = torch.nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            self.convs.append(torch.nn.Conv1d(
                config.embedding.dimension, config.TextGRA.num_kernels,
                kernel_size, padding=kernel_size - 1))

        self.top_k = self.config.TextGRA.top_k_max_pooling
        hidden_size = len(config.TextGRA.kernel_sizes) * \
                      config.TextGRA.num_kernels * self.top_k

        self.rnn = RNN(
            config.embedding.dimension, config.TextGRA.hidden_dimension,
            num_layers=config.TextGRA.num_layers, batch_first=True,
            bidirectional=config.TextGRA.bidirectional,
            rnn_type=config.TextGRA.rnn_type)
        self.rnn2 = RNN(
            600, config.TextGRA.hidden_dimension,
            num_layers=config.TextGRA.num_layers, batch_first=True,
            bidirectional=config.TextGRA.bidirectional,
            rnn_type=config.TextGRA.rnn_type)
        hidden_dimension = config.TextGRA.hidden_dimension
        if config.TextGRA.bidirectional:
            hidden_dimension *= 2

        self.linear_u = torch.nn.Linear(300, 300)

        self.linear = torch.nn.Linear(hidden_dimension, len(dataset.label_map))
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)

    def get_parameter_optimizer_dict(self):
        params = list()
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.char_embedding.parameters()})
        params.append({'params': self.convs.parameters()})
        params.append({'params': self.rnn.parameters()})
        params.append({'params': self.rnn2.parameters()})
        params.append({'params': self.linear_u.parameters()})
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

        # RNN sentence vector
        S, output1= self.rnn(embedding, seq_length) #[batch,n, 300]
        embedding2 = embedding.transpose(1, 2)

        # CNN parase vector
        pooled_outputs = []
        for i, conv in enumerate(self.convs):
            #convolution = torch.nn.ReLU(conv(embedding))
            convolution = torch.nn.functional.relu(conv(embedding2)) #[batch,100,n]

            pooled = torch.topk(convolution, self.top_k)[0].view(
                convolution.size(0), -1)
            pooled_outputs.append(pooled)

        P = torch.cat(pooled_outputs, 1)  ##[batch,300]

        P=P.expand(embedding.size(1),P.size(0),P.size(1)) #[batch,n,300]

        # attention scoring
        attention_score=self.linear_u(torch.tanh(torch.mul(S,P.transpose(0, 1)))) #[batch,n,300]
        # attention gate
        A, output2= self.rnn(attention_score,seq_length)  # [batch,n, 450]
        # attention parase vector
        C=torch.mul(A,P.transpose(0, 1)) #[batch,n,450]
        # combine [x,c] input into RNN
        input = torch.cat((C,embedding), 2)  ##[batch,n,600]

        output, last_hidden = self.rnn2(input, seq_length) #[batch,n,450]
        # average output
        if self.doc_embedding_type == DocEmbeddingType.AVG:
            doc_embedding = torch.sum(output, 1) / seq_length.unsqueeze(1)

        out=self.dropout(self.linear(doc_embedding))

        return out
