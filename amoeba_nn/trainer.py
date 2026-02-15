# -*- coding: utf-8 -*-
# @Time       : 2022/05/18 17:00:52
# @Author     : Yanxing Wang <yxw@utexas.edu>
# @Project    : amoeba_nn
# @Description: trainer for amoeba_nn


from functools import partial
import os
import logging
import pandas as pd
import numpy as np
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .model.utils import get_model, get_loss
from .data.dataset import DatasetSpeedUp, get_dataset, RelativeDataset
from .utils.config import config
from .utils.helpers import set_seed, save, PerformanceSummary


logger = logging.getLogger(__name__)


def train_pipe():
    logger.info("training initializing...")
#    logger.debug(f"current memory usage:\n{os.popen('free -th; df -h').read()}")
#    logger.debug(f"open files info: \n ulimit -n: {os.popen('ulimit -n').read().strip()} \n "
#                f"lsof | wc -l: {os.popen('lsof | wc -l').read().strip()} \n "
#                f"lsof -u yw24267 | wc -l: {os.popen('lsof -u yw24267 | wc -l').read().strip()} \n "
#                f"lsof -u yw24267 | grep \"dev/shm/torch\" | wc -l: {os.popen('lsof -u yw24267 | grep \"dev/shm/torch\" | wc -l').read().strip()} \n "
#                f"cat /proc/sys/fs/file-max: {os.popen('cat /proc/sys/fs/file-max').read().strip()}")
    if config["model"]["pretrained_model"]:
        logger.info(f'finetune model {config["model"]["pretrained_model"]}')
    set_seed()
    torch.multiprocessing.set_sharing_strategy('file_system')
    train_config = config["train"]
    df = pd.read_csv(train_config["csv_path"])
    # df = df[:1000]  # for test and debug
    df_train = df[df[train_config["split"]["column"]].isin(train_config["split"]["train"])]
    df_val = df[df[train_config["split"]["column"]].isin(train_config["split"]["val"])]

    dataset_class = get_dataset(train_config["dataset_name"])
    train_set = dataset_class(
        df_train,
        h5_files=train_config["h5_files"],
        h5_multipole=train_config["h5_multipole"],
        h5_force=train_config["h5_force"],
        label_col=train_config["label_column"],
        device="cpu",
        shuffle=False,  # shuffle with dataloader
        h5_inmemory=False,
    )
    val_set = dataset_class(
        df_val,
        h5_files=train_config["h5_files"],
        h5_multipole=train_config["h5_multipole"],
        h5_force=train_config["h5_force"],
        label_col=train_config["label_column"],
        device="cpu",
        shuffle=False,
        h5_inmemory=False,
    )

    if train_config["relative_training"]:
        train_set = RelativeDataset(train_set)
        val_set = RelativeDataset(val_set)

    logger.debug(f"trainset: {type(train_set)}, len: {len(train_set)}")
    logger.debug(f"valset: {type(val_set)}, len: {len(val_set)}")

    if train_config["speedup"]:
        logger.info("Using New DatasetSpeedUp...")
#        logger.debug(f"current memory usage:\n{os.popen('free -th; df -h').read()}")
#        logger.debug(f"open files info: \n ulimit -n: {os.popen('ulimit -n').read().strip()} \n "
#                    f"lsof | wc -l: {os.popen('lsof | wc -l').read().strip()} \n "
#                    f"lsof -u yw24267 | wc -l: {os.popen('lsof -u yw24267 | wc -l').read().strip()} \n "
#                    f"lsof -u yw24267 | grep \"dev/shm/torch\" | wc -l: {os.popen('lsof -u yw24267 | grep \"dev/shm/torch\" | wc -l').read().strip()} \n "
#                    f"cat /proc/sys/fs/file-max: {os.popen('cat /proc/sys/fs/file-max').read().strip()}")
        train_loader = DatasetSpeedUp(
            train_set, batch_size=train_config["batch_size"], num_workers=train_config["loading_workers"], shuffle=True, use_pkl=train_config["use_pkl"],
        )
        train_loader_for_training = train_loader
        train_loader_for_test = train_loader
        val_loader = DatasetSpeedUp(
            val_set, batch_size=train_config["batch_size"], num_workers=train_config["loading_workers"], shuffle=False, use_pkl=train_config["use_pkl"],
        )
    else:
        train_loader_for_training = DataLoader(
            train_set,
            batch_size=train_config["batch_size"],
            shuffle=True,  
            pin_memory=False,
            num_workers=train_config["loading_workers"],
            collate_fn=train_set.collate_batch,
        )
        train_loader_for_test = DataLoader(
            train_set,
            batch_size=train_config["batch_size"],
            shuffle=False,
            pin_memory=False,
            num_workers=train_config["loading_workers"],
            collate_fn=train_set.collate_batch,
        )

        # train_loader = partial(
        #     DataLoader,
        #     dataset=train_set,
        #     batch_size=train_config["batch_size"],
        #     pin_memory=True,
        #     num_workers=train_config["loading_workers"],
        #     collate_fn=ANINetworkDataset.collate_batch,
        # )
        logger.debug(f"train_loader: {type(train_loader_for_training)}, len: {len(train_loader_for_training)}")

        val_loader = DataLoader(
            val_set,
            batch_size=train_config["batch_size"],
            shuffle=False,
            pin_memory=False,
            num_workers=train_config["loading_workers"],
            collate_fn=val_set.collate_batch,
        )
        logger.debug(f"val_loader: {type(val_loader)}, len: {len(val_loader)}")

    # TODO test sets
    train_size, val_size = len(train_set), len(val_set)
    total_size = train_size + val_size
    logger.info("Size of data:")
    logger.info(f"Train: {train_size}\t({train_size/total_size*100:.1f}%)")
    logger.info(f"Val  : {val_size}\t({val_size/total_size*100:.1f}%)")
#    logger.debug(f"current memory usage:\n{os.popen('free -th; df -h').read()}")
#    logger.debug(f"open files info: \n ulimit -n: {os.popen('ulimit -n').read().strip()} \n "
#                f"lsof | wc -l: {os.popen('lsof | wc -l').read().strip()} \n "
#                f"lsof -u yw24267 | wc -l: {os.popen('lsof -u yw24267 | wc -l').read().strip()} \n "
#                f"lsof -u yw24267 | grep \"dev/shm/torch\" | wc -l: {os.popen('lsof -u yw24267 | grep \"dev/shm/torch\" | wc -l').read().strip()} \n "
#                f"cat /proc/sys/fs/file-max: {os.popen('cat /proc/sys/fs/file-max').read().strip()}")
    
    model = get_model(config["model"]["arch"], model_ckpt=config["model"]["pretrained_model"], device=config["device"])
    logger.debug(str(model))
    logger.debug(f"model device: {next(model.parameters()).device}")
    optimizer = optim.Adam(model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"],)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=train_config["lr_factor"],
        patience=train_config["lr_patience"],
        threshold=train_config["lr_threshold"],
        threshold_mode=train_config["lr_threshold_mode"],
    )
    loss = get_loss(train_config["loss_fn"], train_config.get("sample_weight", None))
    model.set_optimizer(optimizer)
    model.set_scheduler(scheduler)
    model.set_loss(loss)

    train_writer = SummaryWriter(
        os.path.join(config["save_path"], config["save_folder"], "tb", "train"), flush_secs=10
    )  # tb stands for tensorboard.
    val_writer = SummaryWriter(os.path.join(config["save_path"], config["save_folder"], "tb", "val"), flush_secs=10)

    # record the configs
    train_writer.add_text("Runtime_Config", config.to_tensorboard())
    val_writer.add_text("Runtime_Config", config.to_tensorboard())

    # torch.cuda.synchronize()
    logger.info("training started...")
    perfsum = PerformanceSummary()
    for epoch_idx in range(train_config["num_epochs"]):
        logger.info(f"epoch {epoch_idx}: training started.")
        model.train_epoch(
            dataloader=train_loader_for_training, epoch_idx=epoch_idx, writer=train_writer,
        )
        logger.info(f"epoch {epoch_idx}: training completed.")
        perfs_train = model.test_epoch(dataloader=train_loader_for_test, epoch_idx=epoch_idx, writer=train_writer,)
        logger.info(f"epoch {epoch_idx}: train loss: {perfs_train['Loss']:.2f}")
        perfs_val = model.test_epoch(dataloader=val_loader, epoch_idx=epoch_idx, writer=val_writer,)
        logger.info(f"epoch {epoch_idx}: validation loss: {perfs_val['Loss']:.2f}.")
        # logger.info(f"epoch {epoch_idx}: train loss: {loss_train:.2f}, validation loss: {loss_val:.2f}.")
        model.scheduler_step(perfs_val["Loss"], epoch_idx, train_writer)
        save(
            model, epoch_idx, perfs_val["Loss"], os.path.join(config["save_path"], config["save_folder"], "models"),
        )
        perfsum.update(perfs_val=perfs_val, perfs_train=perfs_train)
        if perfsum.last_val["Epoch_ID"] - perfsum.best_val["Epoch_ID"] >= 2 * model.scheduler.patience:
            logger.info(f"Early Stopped at Epoch {epoch_idx}")
            break

    perfsum.to_df().to_csv(os.path.join(config["save_path"], config["save_folder"], "summary.csv"), index=False)
    logger.info(f"Summary:\n{perfsum.to_df()}")
    logger.info("training finished.")
    train_writer.close()
    val_writer.close()
    return perfsum.to_df()


def cv_pipe():
    """ for cross validation"""
    split_config = config["train"]["split"]
    val_splits = np.array_split(split_config["train"], split_config["num_folds"])
    train_splits = []
    for val in val_splits:
        train_splits.append(np.setdiff1d(split_config["train"], val))
    config.update({
        "save_path": os.path.join(config["save_path"], config["save_folder"]),
    })
    cv_results = pd.DataFrame()
    for fold_idx, (train_idx, val_idx) in enumerate(zip(train_splits, val_splits)):
        logger.info("="*20 + f" Fold {fold_idx} started. " + "="*20)
        config.update({
            "train": {
                "split": {
                    "train": list(map(int, train_idx)),
                    "val": list(map(int, val_idx)),
                }
            },
            "save_folder": f"fold_{fold_idx}",
        })
        config.save(os.path.join(config["save_path"], config["save_folder"], "config.yaml"))
        perf_sum_df = train_pipe()
        perf_sum_df["Fold_ID"] = fold_idx
        cv_results = pd.concat([cv_results, perf_sum_df], ignore_index=True)
        logger.info("="*20 + f" Fold {fold_idx} finished. " + "="*20)
    cv_results.to_csv(os.path.join(config["save_path"], "cv_summary.csv"), index=False)
