# -*- coding: utf-8 -*-
# @File: pt2prm.py | amoeba_nn
# @Author: Yanxing Wang <yxw@utexas.edu>
# @Created: February 15 04:06:33 2026 UTC-06:00
# @Description: a utility to convert .pt model checkpoint to .prm file for Tinker9 with native NN

import torch
import os
from rdkit import Chem

from amoeba_nn.utils.config import config
from amoeba_nn.model.utils import get_model
from amoeba_nn.model.mlp import ANINetwork_Metal


###################################################################
# User defined parameters
###################################################################

base_prm_file = "/home/zh6674/project/cunh3/4/mcsample/amoeba/amoeba09-cu.prm"
torch_pt_file = "/home/zh6674/project/cunh3/mp2energy/tzvp11noh/test_20260112-141212/fold_2/models/Checkpoint_Epoch350_3.68_20260112-190928.pt"
new_prm_file = "./amoeba09-nn-cu_test260121.prm"
amoebann_ff_name = "AMOEBA+NN-Cu"

nn_type = "metal"
topo_cutoff = 0
section_delimiter = "\n\n\n"
subsection_delimiter = "\n\n"
num_digits = 16
indent_length = 4
num_prms_per_line = 4

###################################################################
# End of user defined parameters
###################################################################


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
