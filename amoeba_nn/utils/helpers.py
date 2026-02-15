# -*- coding: utf-8 -*-
# @Time       : 2022/05/19 18:24:40
# @Author     : Yanxing Wang <yxw@utexas.edu>
# @Project    : amoeba_nn
# @Description: helper functions and classes


import os
import re
import torch
import logging
import time
import random
import shutil
import subprocess
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import r2_score
from rdkit import Chem
from rdkit.Chem.rdMolTransforms import GetBondLength, GetAngleDeg, GetDihedralDeg
from rdkit.Chem import rdFMCS, rdForceFieldHelpers, AllChem
from itertools import combinations


logger = logging.getLogger(__name__)


def set_seed(seed=97):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def save(model, epoch, loss, model_path):
    t = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))
    torch.save(model.state_dict(), f"{model_path}/Checkpoint_Epoch{epoch}_{loss:.2f}_{t}.pt")
    logger.info(f"Checkpoint Epoch {epoch} Saved at {t}.")


def reset_logger(log_file=None, level=logging.INFO):
    root_logger = logging.getLogger()
    for h in root_logger.handlers:
        root_logger.removeHandler(h)
    basic_fmt = "[%(asctime)s] %(levelname)-8s | %(filename)-12s:%(lineno)-4d | %(message)s"
    logging.basicConfig(
        filename=None, filemode="a", format=basic_fmt, level=level,
    )
    if log_file:
        root_logger = logging.getLogger()
        fhlr = logging.FileHandler(log_file)
        formatter = logging.Formatter(basic_fmt)
        fhlr.setFormatter(formatter)
        root_logger.addHandler(fhlr)


def archive_code(code_path, arch_dir):
    path, folder = os.path.split(code_path)
    os.system(f"cd {path} && tar -zcf {arch_dir}/{folder}.tar.gz {folder}")


class classproperty:
    """Decorator to make a class method a class attribute, just like @property but for class attributes instead of intance attributes."""
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, instance, owner):
        return self.fget(owner)


class Regressor_Performance:
    def __init__(self, ori, pre, threshold=7.0):
        """
        param labels: a list of true labels
        param scores: a list of predict scores
        param threshold: threshold
        """
        self.ori = np.array(ori)
        self.pre = np.array(pre)
        self.threshold = threshold

    def r_square(self):
        """
        Task:    To compute the R-square value
        Input:   ori    Vector with original labels
                pre    Vector with predicted labels
        Output:  r2   R-square value
        """
        r2 = r2_score(self.ori, self.pre)
        return r2

    def rmse(self):
        """
        Task:    To compute root mean squared error (RMSE)
        Input:   ori    Vector with original labels
                pre    Vector with predicted labels
        Output:  rmse_value   RSME
        """
        rmse_value = np.sqrt(((self.ori - self.pre) ** 2).mean(axis=0))
        return rmse_value

    def mae(self):
        """
        Task:    To compute mean squared error (MAE)
        Input:   ori    Vector with original labels
                pre    Vector with predicted labels
        Output:  mae_value   MAE
        """
        mae_value = (np.abs(self.ori - self.pre)).mean(axis=0)
        return mae_value

    def pearson(self):
        """
        Task:    To compute Pearson correlation coefficient
        Input:   ori      Vector with original labels
                pre      Vector with predicted labels
        Output:  pearson_value  Pearson correlation coefficient
        """
        pearson_value = np.corrcoef(self.ori, self.pre)[0, 1]
        return pearson_value

    def spearman(self):
        """
        Task:    To compute Spearman's rank correlation coefficient
        Input:   ori      Vector with original labels
                pre      Vector with predicted labels
        Output:  spearman_value     Spearman's rank correlation coefficient
        """
        spearman_value = stats.spearmanr(self.ori, self.pre)[0]
        return spearman_value

    def get_all(self):
        """
        Task:   Print all evaluation values.
        """

        rmse_value = self.rmse()
        mae = self.mae()
        pearson_value = self.pearson()
        spearman_value = self.spearman()
        r2 = self.r_square()

        return [rmse_value, mae, pearson_value, spearman_value, r2]
    
    def get_all_as_dict(self):
        rmse_value, mae, pearson_value, spearman_value, r2 = self.get_all()
        return {
            "RMSE": rmse_value,
            "MAE": mae,
            "R_Pearson": pearson_value,
            "R_Spearman": spearman_value,
            "R_Squared": r2,
        }
    

@dataclass
class PerformanceSummary:
    best_train = {}
    last_train = {}
    best_val = {}
    last_val = {}
    
    def update(self, perfs_val, perfs_train=None):
        self.last_val = perfs_val
        if self.last_val["Loss"] < self.best_val.get("Loss", float('inf')):
            self.best_val = self.last_val
        if perfs_train:
            self.last_train = perfs_train
            if self.last_train["Loss"] < self.best_train.get("Loss", float('inf')):
                self.best_train = self.last_train
                
    def to_df(self):
        data = [self.best_val, self.last_val, self.best_train, self.last_train]
        remarks = ["Best,Validation", "Last,Validation", "Best,Training", "Last,Training"]
        df = pd.DataFrame()
        for d, r in zip(data, remarks):
            df = pd.concat([df, pd.DataFrame({"Remarks": r, **d}, index=[0])], ignore_index=True)
        return df
