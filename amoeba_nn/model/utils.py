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
from rdkit import Chem

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


def pt2prm(base_prm_file="", torch_pt_file="", new_prm_file="", amoebann_ff_name="", nn_type="", 
           topo_cutoff=0, section_delimiter="\n\n\n", subsection_delimiter="\n\n", 
           num_digits=16, indent_length=4, num_prms_per_line=4):
    # pasre parameters
    with open(base_prm_file) as f:
        base_prms = f.read()
    cont_start = base_prms.index("\n")+1
    cont_end = -base_prms[::-1].index("\n")-1
    head_spacing = base_prms[:cont_start]
    tail_spacing = base_prms[cont_end:]
    prms_sections = base_prms[cont_start:cont_end].split(section_delimiter)
    prms_sections_names = []
    s_name = ''
    for s in prms_sections:
        s = s.strip()
        if s.startswith("######"):
            s_name = s.splitlines()[2].strip("# ")
        prms_sections_names.append(s_name)

    # change the force field name 
    ff_def_sid = prms_sections_names.index('Force Field Definition') + 1
    ff_def = prms_sections[ff_def_sid].splitlines()
    for i, l in enumerate(ff_def):
        if l.strip().startswith("forcefield "):
            ff_def[i] = l[:-l[::-1].index(" ")] + amoebann_ff_name
            break
    prms_sections[ff_def_sid] = "\n".join(ff_def)

    # add neural network parameters
    # 0. load nn model
    config_path = os.path.join(os.path.dirname(os.path.dirname(torch_pt_file)), "config.yaml")
    config.load(config_path)
    model = get_model(config["model"]["arch"], model_ckpt=torch_pt_file, device="cpu")
    species = config["supported_species"]
    pt = Chem.GetPeriodicTable()
    species = [str(pt.GetAtomicNumber(s)) for s in config["supported_species"]]

    if isinstance(model, ANINetwork_Metal):
        species_central = [species[model.metal_species_num]]
        species_neighbor = [s for s in species if s not in species_central]
    else:
        species_central, species_neighbor = species, species

    # 0. header
    nn_prm = f"nnp {nn_type}\n"
    # 1. aev
    # order: R_m_0, R_m_c, R_m_d, eta_m, R_q_0, R_q_c, R_q_d, eta_q, zeta_p, theta_p_d, \n, list of supported atomic numbers
    aev = {}
    for k in ("Rcr", "Rca"):
        aev[k] = model.aev_computer.__getattribute__(k)
    for k in ("EtaR", "ShfR", "EtaA", "Zeta", "ShfA", "ShfZ"):
        aev[k] = model.aev_computer.__getattr__(k).view(-1).tolist()
    aev_line = " " * indent_length + "aev"
    # aev_line += f" {aev['ShfR'][0]:.{num_digits}f}  {aev['Rcr']:.{num_digits}f}  {len(aev['ShfR'])}  {aev['EtaR'][0]:.{num_digits}f}"
    # aev_line += f"  {aev['ShfA'][0]:.{num_digits}f}  {aev['Rca']:.{num_digits}f}  {len(aev['ShfA'])}  {aev['EtaA'][0]:.{num_digits}f}"
    # aev_line += f"  {aev['Zeta'][0]:.{num_digits}f}  {len(aev['ShfZ'])}\n"
    aev_line += f" {aev['ShfR'][0]:.{1}f}  {aev['Rcr']:.{1}f}  {len(aev['ShfR'])}  {aev['EtaR'][0]:.{1}f}"
    aev_line += f"  {aev['ShfA'][0]:.{1}f}  {aev['Rca']:.{1}f}  {len(aev['ShfA'])}  {aev['EtaA'][0]:.{1}f}"
    # aev_line += f"  {aev['Zeta'][0]:.{1}f}  {len(aev['ShfZ'])}\n"
    aev_line += f"  {aev['Zeta'][0]:.{1}f}  {len(aev['ShfZ'])}  {topo_cutoff}\n"
    aev_line += " " * (indent_length + 4) + "  ".join(species_neighbor) + "\n"
    nn_prm += aev_line
    for i, s in enumerate(species_central):
        # 2. general parameters
        nn_prm += " " * indent_length + f"nn {s}\n"
        
        # 3. nn layers
        net_lines = ""
        net = model.potential_networks[i]
        layers = []
        for l in net.network:
            if isinstance(l, torch.nn.Linear):
                net_lines += " " * (indent_length + 3) + "linear"
                weights = l.weight.flatten().tolist()
                weights_shape = l.weight.shape
                biases = l.bias.tolist() if l.bias else [0.0, ] * weights_shape[0]
                for pi, p in enumerate(weights + biases):
                    # net_lines += f" {p:{num_digits*2}.{num_digits}f}"
                    net_lines += f" {p:{num_digits+10}.{num_digits}e}"
                    if (pi + 1) % num_prms_per_line == 0:
                        net_lines += "\n" + " " * (indent_length + 9)
                net_lines = net_lines.rstrip() + "\n"
            elif isinstance(l, torch.nn.CELU):
                # net_lines += " " * (indent_length + 3) + "celu" + f" {l.alpha:{num_digits}.{num_digits}f}\n"
                net_lines += " " * (indent_length + 3) + "celu" + f" {l.alpha:.{6}f}\n"
            else:
                raise NotImplementedError(l)
        nn_prm += net_lines.rstrip() + "\n"

    prms_sections.append(
        "      #################################\n"
        "      ##                             ##\n"
        "      ##  Neural Network Parameters  ##\n"
        "      ##                             ##\n"
        "      #################################"
    )
    prms_sections.append(nn_prm.rstrip())

    # write params to file
    with open(new_prm_file, "w") as f:
        f.write(head_spacing + section_delimiter.join(prms_sections) + tail_spacing)
