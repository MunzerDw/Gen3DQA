import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from model.vqg.positional_encoding_text import PositionalEncodingText
from util.beam_search import BeamSearch, Module
from util.utils import TensorOrNone, generate_square_subsequent_mask

MAX_LEN = 40

class Decoder(Module):
    def __init__(
        self,
        cfg,
        d_model,
        question_vocabulary,
        question_embeddings,
        nhead=6,
        d_hid=300,
        nlayers=2,
        dropout=0.5,
    ):
        super().__init__()
        self.cfg = cfg
        self.question_vocabulary = question_vocabulary
        self.register_buffer("question_embeddings", torch.FloatTensor(question_embeddings))

        # decoder for both modalities
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, d_hid, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, nlayers)

        # map output to distribution over question vocabulary
        self.map_to_vocab = nn.Sequential(
            nn.Linear(300, len(question_vocabulary["token2idx"]))
        )

        # positional encoding for text sequence
        self.positional_encoding_text = PositionalEncodingText(emb_size=300)

        # for greedy decoding
        self.testing = False
        # we use this attribute to determine whether we want to generate
        # greedy sentences for logging the accuracies during training with teacher forcing
        self.validation = False
        # we use this attribute for generating a baseline sample for CIDEr loss
        self.baseline = False

    def forward(
        self,
        batch,
        memory,
        question_embeddings,
        memory_mask,
        question_embeddings_mask,
        question_embeddings_attn_mask,
    ):
        if self.testing:
            seq, seqLogprobs = self._sample(
                memory, memory_mask, greedy=True
            )
            return seq
        elif self.baseline:
            # bs = BeamSearch(self, max_len=MAX_LEN, eos_idx=self.question_vocabulary["token2idx"]["<end>"], beam_size=2)
            # seq, _, seqLogprobs = bs.apply(memory, out_size=2, return_probs=True, memory_mask=memory_mask)
            # batch["seq_sample"] = seq
            # batch["seq_logprobs_sample"] = seqLogprobs
            # return seq
            seq, seqLogprobs = self._sample(
                memory, memory_mask, greedy=False
            )
            batch["seq_sample"] = seq
            return seq
        elif self.cfg.model.activate_cider_loss:
            seq, seqLogprobs = self._sample(
                memory, memory_mask, greedy=True
            )
            batch["seq_greedy"] = seq
            batch["seq_logprobs_greedy"] = seqLogprobs
            return seq
        else:
            if self.validation:
                seq, seqLogprobs = self._sample(
                    memory, memory_mask, greedy=True
                )
                batch["seq_greedy"] = seq
                batch["seq_logprobs_greedy"] = seqLogprobs

            vqg_output = self._tf(
                memory,
                question_embeddings,
                memory_mask,
                question_embeddings_mask,
                question_embeddings_attn_mask,
            )
            return vqg_output

    def _sample(self, memory, memory_mask, greedy=False):
        batch_size = memory.size(0)
        start_idx = self.question_vocabulary["token2idx"]["<start>"]
        end_idx = self.question_vocabulary["token2idx"]["<end>"]

        # we do + 1 because of the start token <start>
        seq = memory.new_zeros(batch_size, MAX_LEN + 1, dtype=torch.long)
        seqLogprobs = memory.new_zeros(
            batch_size, MAX_LEN + 1, len(self.question_vocabulary["token2idx"])
        )

        seq[:, 0] = start_idx

        for t in range(MAX_LEN):
            tgt_mask = generate_square_subsequent_mask(t + 1).to(memory.device)
            seq_embeddings = (
                self.question_embeddings[seq[:, : t + 1].reshape(-1), :].to(memory.device)
            ).reshape(batch_size, t + 1, 300)
            question_embeddings_mask = (seq[:, : t + 1] == 0).bool()
            output = self.decoder(
                self.positional_encoding_text(seq_embeddings).transpose(0, 1),
                memory.transpose(0, 1),
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=question_embeddings_mask,
                memory_key_padding_mask=memory_mask,
            ).transpose(0, 1)

            # map output vectors to vocabulary length
            output = self.map_to_vocab(output)
            output = output[:, -1]
            logprobs = F.log_softmax(output, dim=1)

            if greedy:
                sampleLogprobs, it = torch.max(logprobs.detach(), 1)
            else:
                # fetch prev distribution: shape NxM
                prob_prev = torch.exp(logprobs.detach()).cpu()
                it = torch.multinomial(prob_prev, 1).to(logprobs.device)
                # gather the logprobs at sampled positions
                sampleLogprobs = logprobs.gather(1, it)

            # and flatten indices for downstream processing
            it = it.view(-1).long()

            # stop when all finished
            if t == 0:
                unfinished = it != end_idx
            else:
                it = it * unfinished.type_as(it)
                unfinished = unfinished & (it != end_idx)
            seq[:, t + 1] = it
            seqLogprobs[:, t + 1] = logprobs
            if unfinished.sum() == 0:
                break

        seq = seq[:, 1:]
        seqLogprobs = seqLogprobs[:, 1:, :]

        return seq, seqLogprobs

    # for memory mesh transformer beam search algorithm
    def step(self, t, question_embeddings, memory, **kwargs):
        start_idx = self.question_vocabulary["token2idx"]["<start>"]
        memory_mask = kwargs["memory_mask"]

        if t == 0:
            question_embeddings = (
                torch.Tensor(self.question_embeddings[start_idx].cpu())
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(memory.shape[0], 1, 1)
            ).to(memory)
        else:
            question_embeddings = self.question_embeddings[question_embeddings].to(memory)
            memory = memory.repeat_interleave(
                question_embeddings.shape[0] // memory_mask.shape[0], 0
            )
            memory_mask = memory_mask.repeat_interleave(
                question_embeddings.shape[0] // memory_mask.shape[0], 0
            )

        question_embeddings_attn_mask = generate_square_subsequent_mask(
            question_embeddings.shape[1]
        ).to(memory)

        # shared decoder
        output = self.decoder(
            self.positional_encoding_text(question_embeddings).transpose(0, 1),
            memory.transpose(0, 1),
            tgt_mask=question_embeddings_attn_mask,
            memory_key_padding_mask=memory_mask,
        )

        output = output.transpose(0, 1)

        # map output vectors to vocabulary length
        output = self.map_to_vocab(output)
        output = output[:, -1, :]

        output = F.log_softmax(output, dim=-1)

        return output

    def _tf(
        self,
        memory,
        question_embeddings,
        memory_mask,
        question_embeddings_mask,
        question_embeddings_attn_mask,
    ):
        # shared decoder
        output = self.decoder(
            question_embeddings.transpose(0, 1),
            memory.transpose(0, 1),
            tgt_mask=question_embeddings_attn_mask,
            tgt_key_padding_mask=question_embeddings_mask,
            memory_key_padding_mask=memory_mask,
        )

        output = output.transpose(0, 1)

        # map output vectors to vocabulary length
        output = self.map_to_vocab(output)

        return output
