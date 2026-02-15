# -*- coding: utf-8 -*-
# @Time       : 2022/06/20 17:05:14
# @Author     : Yanxing Wang <yxw@utexas.edu>
# @Project    : amoeba_nn
# @Description: predictor for amoeba_nn


import os
import logging
import pandas as pd
from torch.utils.data import DataLoader

from .model.utils import get_model
from .data.dataset import get_dataset
from .utils.config import config, load_yaml
from .utils.helpers import Regressor_Performance


logger = logging.getLogger(__name__)


def predict_pipe():
    logger.info("prediction initializing...")

    pred_config = config["predict"]
    df = pd.read_csv(pred_config["csv_path"])

    dataset_class = get_dataset(pred_config["dataset_name"])
    dataset = dataset_class(
        df,
        h5_files=pred_config["h5_files"],
        h5_multipole=pred_config["h5_multipole"],
        label_col=pred_config["label_column"],
        device="cpu",
        shuffle=False,  # shuffle with dataloader
        h5_inmemory=True,
    )
    logger.info(f"Size of data: {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=pred_config["batch_size"],
        shuffle=False,
        pin_memory=False,
        num_workers=pred_config["loading_workers"],
        collate_fn=dataset_class.collate_batch,
    )

######
    if pred_config["model_ckpt"] in ('ANI1x', 'ANI1ccx', 'ANI2x'):
        model = get_model(pred_config["model_ckpt"], device=config["device"])
    else:
        model_config = os.path.join(os.path.dirname(os.path.dirname(pred_config["model_ckpt"])), "config.yaml")
        logger.info(f"loading model configuration from {model_config}")
        model_config = load_yaml(model_config)
        config.update({
            k: model_config[k] for k in ["model", "aev", "supported_species"]
        })
        logger.info(f"updated config: {config}")
        model = get_model(config["model"]["arch"], model_ckpt=pred_config["model_ckpt"], device=config["device"])
    logger.debug(f"model device: {next(model.parameters()).device}")

    logger.info("prediction started...")
    preds = model.predict(loader)
    ys = df[pred_config["label_column"]].to_numpy()
    evaluator = Regressor_Performance(ys, preds)
    logger.info(f"RMSE: {evaluator.rmse()}")
    logger.info(f"R_Pearson: {evaluator.pearson()}")
    logger.info(f"R_Spearman: {evaluator.spearman()}")
    logger.info(f"R_squared: {evaluator.r_square()}")

    df[pred_config["pred_column"]] = preds
#####

    save_path = os.path.splitext(os.path.basename(pred_config["csv_path"]))[0] + '_pred.csv'
    save_path = os.path.join(config["save_path"], config["save_folder"], save_path)
    df.to_csv(save_path, index=False)
    logger.info("prediction finished.")
