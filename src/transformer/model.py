import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerClassifier(nn.Module):
    '''
    TransformerEncoder model that takes already vectorized sentences as input.
    Performs binary classification.
    '''

    def __init__(self, num_features=300, num_classes=2, num_layers=4, nhead=6, dim_feedforward=512):
        super().__init__()

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=num_features,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True
            ),
            num_layers=num_layers,
        )

        self.classifier = nn.Linear(num_features, num_classes)


    def forward(self, x, lengths, device="cpu"):

        mask = self.create_padding_mask(lengths, x.shape[1], device)
        mask_float = (~mask).unsqueeze(-1)

        x = self.encoder(x, src_key_padding_mask=mask)
        pooled = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1) # Masked mean pooling by Copilot
        logits = self.classifier(pooled)
        
        return logits


    # By Copilot
    def create_padding_mask(self, lengths, max_len, device):
        return (
            torch.arange(max_len, device=device)
            .expand(len(lengths), max_len)
            >= lengths.unsqueeze(1)
        )