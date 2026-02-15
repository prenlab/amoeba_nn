# -*- coding: utf-8 -*-
# @Time       : 2022/05/19 18:47:14
# @Author     : Yanxing Wang <yxw@utexas.edu>
# @Project    : amoeba_nn
# @Description: main entry of the program.


import argparse
import time
import os
import logging

from .trainer import train_pipe, cv_pipe
from .predictor import predict_pipe
from .utils.config import config
from .utils.helpers import reset_logger, archive_code

logger = logging.getLogger(__name__)


def main():
    t0 = time.time()
    args = get_arguments()
    logging_level = logging.DEBUG if args.debug else logging.INFO
    config.load(args.input_params)
    config.update({"save_folder": f"{config['exp_name']}_{config['time_stamp']}"})

    if args.task == "train":
        os.makedirs(os.path.join(config["save_path"], config["save_folder"], "models"))
    elif args.task == "cv":
        for i in range(config["train"]["split"]["num_folds"]):
            os.makedirs(os.path.join(config["save_path"], config["save_folder"], f"fold_{i}", "models"))
    elif args.task == "predict":
        os.makedirs(os.path.join(config["save_path"], config["save_folder"]))

    log_file = (
        os.path.join(config["save_path"], config["save_folder"], "log") if not args.nolog else None
    )
    reset_logger(level=logging_level, log_file=log_file)
    myhost = os.uname()
    logger.info(f"Running on {myhost.nodename} [{myhost.sysname} {myhost.release}]")

    if args.device:
        config.update({"device": args.device})
    if config["device"] == "cuda":
        config.update({"device": "cuda:0"})
    logger.info(f"Device: {config['device']}")
    # if "cuda" in config["device"]:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = config["device"].split(":")[-1]
    # logger.info(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    config.save(os.path.join(config["save_path"], config["save_folder"], "config.yaml"))
    archive_code(
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(config["save_path"], config["save_folder"]),
    )
    if args.task == "train":
        train_pipe()
    elif args.task == "cv":
        cv_pipe()
    elif args.task == "predict":
        predict_pipe()
    else:
        raise NotImplementedError

    logger.info(f"program ended. total time eclipsed: {time.time() - t0}")


def get_arguments():
    parser = argparse.ArgumentParser(description="AMOEBA+NN")
    parser.add_argument("task", type=str, help="what task to run. options: train, cv, predict.")
    parser.add_argument("-ip", "--input_params", type=str, required=True, help="input yaml file")
    parser.add_argument("--device", type=str, default="", help="device. e.g., cpu, cuda:0.")
    parser.add_argument("--debug", action="store_true", help="logging with debug level")
    parser.add_argument("--nolog", action="store_true", help="disable logging to file")
    return parser.parse_args()
