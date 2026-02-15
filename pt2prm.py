# -*- coding: utf-8 -*-
# @File: pt2prm.py | amoeba_nn
# @Author: Yanxing Wang <yxw@utexas.edu>
# @Created: February 15 04:06:33 2026 UTC-06:00
# @Description: a utility to convert .pt model checkpoint to .prm file for Tinker9 with native NN

import argparse
from amoeba_nn.model.utils import pt2prm


DEFAULT_PARAMS = {
    "base_prm_file": "./amoeba09-cu.prm",
    "torch_pt_file": "./test_20260112-141212/fold_2/models/Checkpoint_Epoch350_3.68_20260112-190928.pt",
    "new_prm_file": "./amoeba09-nn-cu_test260121.prm",
    "amoebann_ff_name": "AMOEBA+NN-Cu",
    "nn_type": "metal",
    "topo_cutoff": 0,
    "section_delimiter": "\n\n\n",
    "subsection_delimiter": "\n\n",
    "num_digits": 16,
    "indent_length": 4,
    "num_prms_per_line": 4,
}


def get_arguments():
    parser = argparse.ArgumentParser(description="Convert PyTorch model checkpoint to Tinker9 PRM file")
    parser.add_argument("--base_prm", type=str, default=DEFAULT_PARAMS["base_prm_file"],
                        help="Path to base .prm file")
    parser.add_argument("--pt_file", type=str, default=DEFAULT_PARAMS["torch_pt_file"],
                        help="Path to PyTorch checkpoint file")
    parser.add_argument("--out_prm", type=str, default=DEFAULT_PARAMS["new_prm_file"],
                        help="Output .prm file path")
    parser.add_argument("--ff_name", type=str, default=DEFAULT_PARAMS["amoebann_ff_name"],
                        help="Force field name")
    parser.add_argument("--nn_type", type=str, default=DEFAULT_PARAMS["nn_type"],
                        help="NN type (e.g., metal)")
    parser.add_argument("--topo_cutoff", type=int, default=DEFAULT_PARAMS["topo_cutoff"],
                        help="Topology cutoff")
    parser.add_argument("--num_digits", type=int, default=DEFAULT_PARAMS["num_digits"],
                        help="Number of digits for parameters")
    parser.add_argument("--indent", type=int, default=DEFAULT_PARAMS["indent_length"],
                        help="Indent length")
    parser.add_argument("--prms_per_line", type=int, default=DEFAULT_PARAMS["num_prms_per_line"],
                        help="Number of parameters per line")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    pt2prm(
        base_prm_file=args.base_prm,
        torch_pt_file=args.pt_file,
        new_prm_file=args.out_prm,
        amoebann_ff_name=args.ff_name,
        nn_type=args.nn_type,
        topo_cutoff=args.topo_cutoff,
        section_delimiter=DEFAULT_PARAMS["section_delimiter"],
        subsection_delimiter=DEFAULT_PARAMS["subsection_delimiter"],
        num_digits=args.num_digits,
        indent_length=args.indent,
        num_prms_per_line=args.prms_per_line,
    )
