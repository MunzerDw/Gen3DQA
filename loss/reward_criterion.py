import torch
import torch.nn as nn


class RewardCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, seq, reward):
        seqlogprobs = input.gather(3, seq.unsqueeze(3)).squeeze(3)

        per_word_reward = (
            reward.unsqueeze(1)
            .unsqueeze(1)
            .repeat(1, seqlogprobs.shape[1], seqlogprobs.shape[2])
        )

        mask = seq > 0

        per_word_loss = -per_word_reward * seqlogprobs * mask

        output = per_word_loss.sum() / mask.sum()

        return output
        # N, L = input.shape[:2]
        # input = input.gather(2, seq.unsqueeze(2)).squeeze(2)

        # input = input.reshape(-1)
        # reward = reward.reshape(-1)
        # mask = seq > 0
        # mask = mask.reshape(-1)
        # # mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(
        # #     -1
        # # )
        # output = -input * reward * mask

        # if reduction == "none":
        #     output = output.view(N, L).sum(1) / mask.view(N, L).sum(1)
        # elif reduction == "mean":
        #     output = torch.sum(output) / torch.sum(mask)

        # return output
