import os
from importlib import import_module

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from callback import *
from data.data_module import DataModule


def filter_weights(ckpt, layers):
    to_delete = []
    for x in ckpt["state_dict"]:
        if x.split(".")[0] not in layers:
            to_delete.append(x)
    while (len(to_delete) > 0):
        del ckpt["state_dict"][to_delete[0]]
        to_delete.pop(0)

def init_model(cfg, train_dataset, testing):
    model = getattr(import_module("model.ours"), "Ours")(cfg, train_dataset, testing)
    if cfg.model.vqg.weights != False:
        ckpt = torch.load(cfg.model.vqg.weights)
        filter_weights(ckpt, ["vqg"])
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print("Loaded VQG model weights")
    if cfg.model.vqa.weights != False:
        ckpt = torch.load(cfg.model.vqa.weights)
        filter_weights(ckpt, ["vqa", "op_map"])
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print("Loaded VQA model weights")
    if cfg.model.softgroup.weights != False:
        ckpt = torch.load(cfg.model.softgroup.weights)
        model.softgroup.load_state_dict(ckpt["state_dict"], strict=True)
        print("Loaded SoftGroup model weights")
    return model


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    # fix the seed
    pl.seed_everything(cfg.global_train_seed, workers=True)

    print("==> initializing data ...")
    data_module = DataModule(cfg)

    print("==> initializing trainer ...")
    trainer = pl.Trainer(logger=None, **cfg.model.trainer)

    print("==> initializing model ...")
    model = init_model(cfg, data_module.vocabularies, testing=True)

    print("==> start testing ...")
    trainer.test(model=model, datamodule=data_module, ckpt_path=cfg.model.ckpt_path)


if __name__ == "__main__":
    main()
