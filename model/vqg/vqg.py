import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vqg.decoder import Decoder
from model.vqg.encoder import Encoder
from model.vqg.positional_encoding_text import PositionalEncodingText
from util.utils import generate_mask, generate_square_subsequent_mask, get_ious


class VQG(nn.Module):
    def __init__(self, cfg, question_vocabulary, question_embeddings):
        super().__init__()
        self.cfg = cfg
        self.question_vocabulary = question_vocabulary
        self.register_buffer("question_embeddings", torch.FloatTensor(question_embeddings))

        # positional encoding for text sequence
        self.positional_encoding_text = PositionalEncodingText(emb_size=300)

        # cross-attention to concatenation encoder
        self.encoder = Encoder(cfg, d_model=300, nhead=6, d_hid=300, nlayers=2)

        # decoder (answer generation)
        self.decoder = Decoder(
            cfg,
            d_model=300,
            nhead=6,
            d_hid=300,
            nlayers=2,
            question_vocabulary=question_vocabulary,
            question_embeddings=question_embeddings,
        )

        # type embeddings
        # 0: object, 1: text
        self.type_embeddings = nn.Embedding(2, 300)

        # target object embeddings
        # 0: target object, 1: not target object
        self.target_embeddings = nn.Embedding(2, 300)

        # object confidence scores
        # self.obj_conf = nn.Sequential(
        #     nn.Linear(300, 300), nn.ReLU(), nn.Dropout(0.1), nn.Linear(300, 1)
        # )
        
    def forward(self, batch):
        question_aware_object_proposals = batch["object_proposals"]
        question_aware_object_proposals_mask = batch["object_proposals_mask"]

        # answer embeddings
        answer_embeddings = batch["answer_embeddings"]
        answer_embeddings_mask = generate_mask(answer_embeddings)
        batch["answer_embeddings_mask"] = answer_embeddings_mask

        # positional encoding for question
        answer_embeddings = self.positional_encoding_text(answer_embeddings).float()
        
        # positional embeddings for object proposals
        question_aware_object_proposals[:, :, :3] = (
            question_aware_object_proposals[:, :, :3] + batch["pred_bboxes_avgs"]
        )
        
        # add type embeddings
        question_aware_object_proposals += self.type_embeddings(
            torch.tensor(0, device=question_aware_object_proposals.device)
        )
        answer_embeddings += self.type_embeddings(
            torch.tensor(1, device=question_aware_object_proposals.device)
        )

        # encode
        memory = self.encoder(
            question_aware_object_proposals.float(),
            question_aware_object_proposals_mask,
            answer_embeddings.float(),
            answer_embeddings_mask,
        )
        batch["vqg_memory"] = memory
        memory_mask = torch.cat(
            [question_aware_object_proposals_mask, answer_embeddings_mask], dim=1
        )
        batch["vqg_memory_mask"] = memory_mask
    
        # add target object embedding to the corresponding object proposal
        self._add_target_object_embedding(batch, memory)

        # prepare decoder target (answer)
        if not self.decoder.testing:
            # hide <end> from target
            question_embeddings = batch["question_embeddings"].float()
            
            non_end_tokens = (
                batch["question_indicies"] != self.question_vocabulary["token2idx"]["<end>"]
            )
            
            question_embeddings = question_embeddings[non_end_tokens].reshape(
                non_end_tokens.shape[0], non_end_tokens.shape[1] - 1, 300
            )

            question_embeddings_mask = generate_mask(question_embeddings)

            batch["question_embeddings_mask"] = question_embeddings_mask

            question_embeddings = self.positional_encoding_text(question_embeddings).float()

            question_embeddings_attn_mask = generate_square_subsequent_mask(
                question_embeddings.shape[1]
            ).to(memory.device)
        else:
            # prediction mode
            question_embeddings = None
            question_embeddings_mask = None
            question_embeddings_attn_mask = None

        # decode
        output = self.decoder(
            batch,
            memory,
            question_embeddings,
            memory_mask,
            question_embeddings_mask,
            question_embeddings_attn_mask,
        )

        batch["vqg_output"] = output

        return output
    
    def _add_target_object_embedding(self, batch, memory):
        if "obj_conf" not in batch:
            for batch_idx in range(batch["object_proposals"].shape[0]):
                batch_idx_object_proposals_mask = ~batch["object_proposals_mask"][
                    batch_idx
                ]
                pred_bboxes = batch["pred_bboxes"]
                batch_idx_pred_bboxes = pred_bboxes[batch_idx][
                    batch_idx_object_proposals_mask
                ]

                gt_bbox = batch["gt_bbox"][batch_idx]
                gt_bbox_repeated = gt_bbox.unsqueeze(0).repeat(
                    batch_idx_pred_bboxes.shape[0], 1
                )

                ious = get_ious(
                    gt_bbox_repeated.cpu().numpy(), batch_idx_pred_bboxes.cpu().numpy()
                )

                # hide ious of bboxes that are background objects
                batch_idx_cls_scores_bg = (
                    np.array(batch["pred_cls_scores"][batch_idx]) == 18
                ).tolist()
                ious[batch_idx_cls_scores_bg] = 0.0

                # we take the predicted bounding box that has the highest iou to the ground truth as the bounding
                # box we need our model to give the highest confidence to
                max_idx = np.argmax(ious)
                memory[batch_idx][max_idx] += self.target_embeddings(
                    torch.tensor(0, device=memory.device)
                )

                batch_idx_object_proposals_mask[max_idx] = False

                # add "not target object" embedding to object proposals
                # that are not the target object
                memory[batch_idx][: batch["object_proposals"].shape[1]][
                    batch_idx_object_proposals_mask
                ] += self.target_embeddings(torch.tensor(1, device=memory.device))
            
        else:
            for batch_idx in range(batch["obj_conf"].shape[0]):
                batch_idx_object_proposals_mask = ~batch["object_proposals_mask"][batch_idx]

                # actual prediction with highest confidence
                # batch_idx_obj_idx = torch.max(
                #     batch["obj_conf"][batch_idx][batch_idx_object_proposals_mask], dim=0
                # ).indices
                k = 1
                k = min(
                    [
                        k,
                        batch["obj_conf"][batch_idx][batch_idx_object_proposals_mask].shape[
                            0
                        ],
                    ]
                )
                batch_idx_obj_idx = torch.topk(
                    batch["obj_conf"][batch_idx][batch_idx_object_proposals_mask], k, dim=0
                ).indices

                batch_idx_object_proposals_mask[batch_idx_obj_idx] = False

                # add "target object" embedding to the target object proposal
                memory[batch_idx][batch_idx_obj_idx] += self.target_embeddings(
                    torch.tensor(0, device=memory.device)
                )

                # add "not target object" embedding to object proposals
                # that are not the target object
                memory[batch_idx][: batch["object_proposals"].shape[1]][
                    batch_idx_object_proposals_mask
                ] += self.target_embeddings(torch.tensor(1, device=memory.device))