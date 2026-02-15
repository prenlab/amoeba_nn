# -*- coding: utf-8 -*-
# @Time       : 2022/07/13 20:20:58
# @Author     : Yanxing Wang <yxw@utexas.edu>
# @Project    : amoeba_nn
# @Description: utils for model manipulation and training

import json
import os
import numpy as np
import torch
from torch import nn
from functools import partial

import logging

from ..utils.config import config
from .mlp import ANINetwork, ANINetwork_pos, ANINetwork_BN, ANINetwork_Multipole, ANINetwork_MonoDipole, ANINetwork_Relative, ANINetwork_Metal, ANINetwork_Metal_Relative
from .ani import ANIWrapper

logger = logging.getLogger(__name__)


def get_model(model_type, model_ckpt=None, device="cpu"):
    """get model

    Args:
        model_type (str): model type
        device (str, optional): torch.device. Defaults to 'cpu'.

    Returns:
        nn.Module: model instance
    """
    force_weight = config.get("train", {}).get("loss_force_weight", 0)
    is_relative = config.get("train", {}).get("relative_training", False)
    if is_relative:
        model_type += "_Relative"
    model_dict = {
        "ANINetwork": partial(ANINetwork, force_weight=force_weight),
        "ANINetwork_Relative": partial(ANINetwork_Relative, force_weight=force_weight),
        "ANINetwork_pos": ANINetwork_pos,
        "ANINetwork_BN": ANINetwork_BN,
        "ANINetwork_Multipole": ANINetwork_Multipole,
        "ANINetwork_MonoDipole": ANINetwork_MonoDipole,
        "ANINetwork_Metal": partial(ANINetwork_Metal, force_weight=force_weight),
        "ANINetwork_Metal_Relative": partial(ANINetwork_Metal_Relative, force_weight=force_weight),
        "ANI1x": partial(ANIWrapper, model_type),
        "ANI1ccx": partial(ANIWrapper, model_type),
        "ANI2x": partial(ANIWrapper, model_type),
    }
    model = model_dict[model_type]().to(device)
    if model_ckpt:
        model.load_state_dict(torch.load(model_ckpt, map_location=device))
        # for param in model.features.parameters():
        #     param.requires_grad = False
    return model


def get_loss(name, sample_weight=None):
    if not sample_weight:
        loss_dict = {
            "MSE": weighted_loss,
            "M4E": partial(weighted_loss, power=4),
            "M8E": partial(weighted_loss, power=8),
            "SquaredRelative": relative_loss,
        }
        return loss_dict[name]
    else:
        return weighted_loss
        # loss_dict = {
        #     "<20_2": 
        # }
        # return loss_dict[sample_weight]


# def high_order_loss(output, target, power):
#     loss = torch.mean((output - target) ** power)
#     return loss


def relative_loss(output, target, weight=1):
    loss = torch.mean(weight * (((output - target) / target) ** 2))
    return loss


def weighted_loss(output, target, weight=1, power=2, for_forces=None, reduction=True):
    # weight = target < 20
    if not for_forces:
        loss = torch.mean(weight * ((output - target) ** power))
    elif for_forces == "CartMSE":     # MSE of Cartesian Coordinates
        weight = weight.unsqueeze(-1).unsqueeze(-1)
        denominator = torch.sum(target != 0).item()
        logger.debug(f"target.shape: {target.shape}, output.shape: {output.shape} weight.shape: {weight.shape}, number of values before padding: {denominator}")
        loss = torch.sum(weight * ((output - target) ** power))
        if not reduction:
            return loss, denominator
        loss = loss / denominator
    elif for_forces == "CosMag":     # Cosien distance and MSE of magnitudes
        weight = weight.unsqueeze(-1)
        denominator = torch.sum(target != 0).item() / 3
        # cosine distance between force vectors
        cos_distance = torch.sum(torch.nan_to_num(weight * (1 - (output * target).sum(-1) / (output ** 2).sum(-1) ** 0.5 / (target ** 2).sum(-1) ** 0.5)))
        # magnitude difference between force vectors, abs error.
        mag_distance = torch.sum(weight * (((output ** 2).sum(-1) ** 0.5 - (target ** 2).sum(-1) ** 0.5) ** 2))
        loss = (cos_distance * config["train"]["loss_CosMag_cos_weight"] + mag_distance) / 2
        logger.debug(f"cos_distance: {cos_distance}, mag_distance: {mag_distance}, loss: {loss}, denominator: {denominator}")
        if not reduction:
            return loss, denominator
        loss = loss / denominator
    return loss


def convert_model_to_tinkerhp_json(model_path, json_path):
    """ convert my model state_dict pt file to json file that works with tinkerhp.
    """
    tkhp_mjson = {
        "rmse": 0.00, 
        "self_energies": [0.0, 0.0, 0.0, 0.0],
        "network_type": "ANIMODEL",
    }

    config_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), "config.yaml")
    config.load(config_path)

    tkhp_mjson["species"] = config["supported_species"]

    model = get_model(config["model"]["arch"], model_ckpt=model_path, device="cpu")

    aev = {}
    for k in ("Rcr", "Rca"):
        aev[k] = model.aev_computer.__getattribute__(k)
    for k in ("EtaR", "ShfR", "EtaA", "Zeta", "ShfA", "ShfZ"):
        aev[k] = model.aev_computer.__getattr__(k).view(-1).tolist()
    tkhp_mjson["aev"] = aev

    network_setup = {}
    for i, net in enumerate(model.potential_networks):
        ele = tkhp_mjson["species"][i]
        layers = []
        for l in net.network:
            if isinstance(l, nn.Linear):
                layers.append({
                    "type": "linear",
                    "nodes": l.bias.shape[0],
                    "weights": l.weight.tolist(),
                    "bias": l.bias.tolist(),
                })
            else:
                assert "activation" not in layers[-1], "Activation Function Already Exists!"
                layers[-1]["activation"] = type(l).__name__
        network_setup[ele] = {"layers": layers}
    tkhp_mjson["network_setup"] = network_setup
    
    open(json_path, "w").write(json.dumps(tkhp_mjson))


def get_ckpt_model(model_ckpt, device="cuda"):
    """get model from checkpoint file, config automatically detected"""
    model_config = os.path.join(os.path.dirname(os.path.dirname(model_ckpt)), "config.yaml")
    config.load(model_config)
    model = get_model(config["model"]["arch"], model_ckpt=model_ckpt, device=device)
    return model


def average_models(ckpts, device="cuda"):
    """average model parameters from checkpoints"""
    models = [get_ckpt_model(model_ckpt=ckpt, device=device) for ckpt in ckpts]
    state_dicts = [model.state_dict() for model in models]
    new_state_dict = {}
    for key in state_dicts[0]:
        new_state_dict[key] = sum([sd[key] for sd in state_dicts]) / len(state_dicts)
    new_model = models[0]
    new_model.load_state_dict(new_state_dict)
    return new_model