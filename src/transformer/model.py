"""
BSD 3-Clause License

Copyright (c) 2017, Pytorch contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Available at: https://github.com/pytorch/examples/blob/main/word_language_model/model.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# From https://github.com/pytorch/examples/blob/main/word_language_model/model.py without modifications
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# Adapted from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
class TransformerClassifier(nn.Transformer):

    def __init__(self, num_features=300, num_classes=2, vocab_size=1000, nhead=6, nhid=512, nlayers=4, dropout=0.1):
        super(TransformerClassifier, self).__init__(d_model=num_features, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers, batch_first=True)
        self.pos_encoder = PositionalEncoding(num_features, dropout)

        self.input_emb = nn.Embedding(vocab_size, num_features)
        self.num_features = num_features
        self.classifier = nn.Linear(num_features, num_classes)

        self.cls_token = nn.Parameter(torch.randn(1, 1, num_features)) # Idea from Copilot

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.classifier.bias)
        nn.init.uniform_(self.classifier.weight, -initrange, initrange)

    def forward(self, src, mask):
        batch_size = src.shape[0]

        # Concatenating the mask to fit the class tokens.
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=src.device)
        mask = torch.cat([cls_mask, mask], dim=1)

        src = self.input_emb(src) * math.sqrt(self.num_features)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        src = torch.cat([cls_tokens, src], dim=1)
        src = self.pos_encoder(src)
        output = self.encoder(src, src_key_padding_mask=mask)
        output = self.classifier(output[:, 0]) # Input only class tokens into the classifier.
        return output
