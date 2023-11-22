import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, cfg, d_model, nhead=6, d_hid=300, nlayers=2, dropout=0.5):
        super().__init__()
        self.cfg = cfg

        # encoder for both modalities
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)

    def forward(
        self,
        object_proposals,
        object_proposals_mask,
        question_embeddings,
        question_embeddings_mask,
    ):

        # shared encoder
        seq = torch.cat([object_proposals, question_embeddings], dim=1).float()
        mask = torch.cat([object_proposals_mask, question_embeddings_mask], dim=1)
        output = self.encoder(seq.transpose(0, 1), src_key_padding_mask=mask)

        output = output.transpose(0, 1)

        return output
