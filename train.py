import os
from importlib import import_module

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from callback import *
from data.data_module import DataModule


def init_callbacks(cfg, output_path):
    val_word_accuracy_checkpoint_monitor = ModelCheckpoint(
        dirpath=os.path.join(output_path, "word_acc_ckpt"),
        monitor="word_accuracy/val",
        mode="max",
        save_top_k=1,
        every_n_epochs=1,
    )
    val_cider_checkpoint_monitor = ModelCheckpoint(
        dirpath=os.path.join(output_path, "cider_ckpt"),
        monitor="CIDEr/val",
        mode="max",
        save_top_k=1,
        every_n_epochs=1,
    )
    val_vqa_loss_checkpoint_monitor = ModelCheckpoint(
        dirpath=os.path.join(output_path, "vqa_loss"),
        monitor="vqa_loss/val",
        mode="min",
        save_top_k=1,
        every_n_epochs=1,
    )
    val_vqg_loss_checkpoint_monitor = ModelCheckpoint(
        dirpath=os.path.join(output_path, "vqg_loss"),
        monitor="vqg_loss/val",
        mode="min",
        save_top_k=1,
        every_n_epochs=1,
    )
    if cfg.model.activate_cider_loss:
        val_vqa_loss_checkpoint_monitor = ModelCheckpoint(
            dirpath=os.path.join(output_path, "vqa_cider_loss"),
            monitor="vqa_cider_loss/val",
            mode="min",
            save_top_k=1,
            every_n_epochs=1,
        )
        val_vqg_loss_checkpoint_monitor = ModelCheckpoint(
            dirpath=os.path.join(output_path, "vqg_cider_loss"),
            monitor="vqg_cider_loss/val",
            mode="min",
            save_top_k=1,
            every_n_epochs=1,
        )
    latest_checkpoint_monitor = ModelCheckpoint(
        dirpath=os.path.join(output_path, "latest"),
        mode="max",
        save_last=False,
        save_top_k=-1,
        every_n_epochs=10,
    )
    gpu_cache_clean_monitor = GPUCacheCleanCallback()
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    time_logging_callback = TimeLoggingCallback()

    result = [
        latest_checkpoint_monitor,
        val_word_accuracy_checkpoint_monitor,
        gpu_cache_clean_monitor,
        lr_monitor,
        time_logging_callback,
        val_cider_checkpoint_monitor,
    ]

    if not cfg.model.freeze_vqa:
        result.append(val_vqa_loss_checkpoint_monitor)

    if not cfg.model.freeze_vqg:
        result.append(val_vqg_loss_checkpoint_monitor)

    return result


def filter_weights(ckpt, layers):
    to_delete = []
    for x in ckpt["state_dict"]:
        if x.split(".")[0] not in layers:
            to_delete.append(x)
    while len(to_delete) > 0:
        del ckpt["state_dict"][to_delete[0]]
        to_delete.pop(0)


def init_model(cfg, train_dataset):
    model = getattr(import_module("model.ours"), "Ours")(cfg, train_dataset)
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

    if cfg.model.precompute_softgroup_data:
        cfg.model.trainer.check_val_every_n_epoch = 1
        cfg.model.trainer.max_epochs = 1
        cfg.model.trainer.num_sanity_val_steps = 0

    output_path = os.path.join(
        cfg.exp_output_root_path, cfg.model.experiment_name, "training"
    )
    os.makedirs(output_path, exist_ok=True)

    print("==> initializing data ...")
    data_module = DataModule(cfg)

    print("==> initializing logger ...")
    logger = getattr(import_module("pytorch_lightning.loggers"), cfg.model.log.module)(
        save_dir=output_path, **cfg.model.log[cfg.model.log.module]
    )

    print("==> initializing monitor ...")
    callbacks = init_callbacks(
        cfg,
        os.path.join(logger.root_dir, "version_" + str(logger.version), "checkpoints"),
    )
    if cfg.model.precompute_softgroup_data:
        callbacks = None

    print("==> initializing trainer ...")
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **cfg.model.trainer)

    print("==> initializing model ...")
    model = init_model(cfg, data_module.vocabularies)

    print("==> start training ...")
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.model.ckpt_path)


if __name__ == "__main__":
    main()
