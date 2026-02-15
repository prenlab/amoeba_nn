# -*- coding: utf-8 -*-
# @Time       : 2023/06/12 18:25:14
# @Author     : Yanxing Wang <yxw@utexas.edu>
# @Project    : amoeba_nn
# @Description: interface for embeding the python implementation of nn into tinker9

import os
import re
import time
import shutil
import torch
import subprocess
import numpy as np
from copy import deepcopy
from itertools import zip_longest

from ..data.dataset import ANINetworkDataset
from .config import config, load_yaml
from ..model.utils import get_ckpt_model


class TinkerXYZ:
    @staticmethod
    def _is_box_info(line):
        for c in line.strip():
            if c.isalpha():
                return False
        else:
            return True
        
    @staticmethod
    def split_line(line):
        """Split a line in txyz file into different parts"""
        # strict mode: all parts are separated by spaces, or '-' at least.
        loose_mode = False
        patt = [
            "^\s*(\d+)\s+([A-Za-z\+\-]+)",  # for atom idx and symbol
            ''.join(["\s*((?:\s|\-)\d+\.\d+(?:e[\-\+0-9][0-9]*)?)",] * 3),  # for coordinates
            ''.join(["(?:\s+(\d+))?",] * 9),  # for atom type + up to 8 connected atoms.
        ]
        line_ = re.findall(''.join(patt), line)
        if not line_:
            # loose mode: use number of decimal digits to split coordinates.
            # find the number of decimal digits of one well separated coordinate string
            patt_decimal = r"(?:\s|\-)\d+\.(\d+)(?:\s|\-)"
            decimal = re.findall(patt_decimal, line)
            if decimal:
                num_digits = len(decimal[0])
                patt[1] = ''.join(["\s*(\-?\d+\.\d{," + str(num_digits) + "})",] * 3)  # for coordinates
                line_ = re.findall(''.join(patt), line)
                loose_mode = True
        if not line_:
            # failed, return None
            return
        line_ = [l for l in line_[0] if l]
        if loose_mode:
            print(f"Warning: Loose mode was used to split the line: '{line}' -> {line_}. Please double check.")
        return line_
        

class Tinker:
    """Python wrapper to call tinker cmdline program
    """
    def __init__(self, wd='.', tinker_path="/home/yw24267/programs/tinker/Tinker8/latest/bin/", timeout=300):
        self.wd = wd
        self.tinker_path = tinker_path
        self.timeout = timeout
    
    def call(self, program, cmd_args='', inter_inps='', envs='', pre_cmds=''):
        """example args: 
        program='analyze'
        cmd_args='xxx.xyz -k xxxx.key' or 'xxx.xyz -k xxxx.key EP'
        inter_inps='EP\\n' or ''
        
        program='protein'
        cmd_args=''
        inter_inps='ala-ala\\n\\n/home/yw24267/programs/tinker/Tinker8/2209/params/amoebabio18.prm\\nACE\\nALA\\nALA\\nNME\\n\\n\\n'

        envs='PYTHONPATH=$PYTHONPATH:/work/yw24267/DLFF/tinker9/ext/'
        """
        pre_cmds = "&& " + pre_cmds if pre_cmds else ''
        try:
            proc = subprocess.Popen(
                f"cd {self.wd} {pre_cmds} && {envs} {self.tinker_path}/{program} {cmd_args}",
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            outs, errs = map(lambda x: x.decode(), proc.communicate(input=inter_inps.encode(), timeout=self.timeout))
        except Exception as e:
            outs, errs = "", str(e)
        finally:
            proc.kill()  # this is important to cleanup properly, e.g., release memory.
        return outs, errs


class AMOEBACalculator:
    """calculate AMOEBA energies.
    """

    energy_names = [
        "AMOEBA_TotPotEne",
        "AMOEBA_BondStretch",
        "AMOEBA_AngleBend",
        "AMOEBA_StretchBend",
        "AMOEBA_OOPBend",
        "AMOEBA_TorsAngle",
        "AMOEBA_PiOrbTors",
        "AMOEBA_VdW",
        "AMOEBA_AtomMultipol",
        "AMOEBA_Polar",
        "AMOEBA_NNValence",
        "AMOEBA_NNMetal",
        "AMOEBA_NNTerm",
        "AMOEBA_GeoRests",
        "AMOEBA_UreyBrad",
    ]
    energy_patts = [
        r"Total Potential Energy :\s*(\-?\d*\.\d*)",
        r"Bond Stretching\s*(\-?\d*\.\d*)",
        r"Angle Bending\s*(\-?\d*\.\d*)",
        r"Stretch-Bend\s*(\-?\d*\.\d*)",
        r"Out-of-Plane Bend\s*(\-?\d*\.\d*)",
        r"Torsional Angle\s*(\-?\d*\.\d*)",
        r"Pi-Orbital Torsion\s*(\-?\d*\.\d*)",
        r"Van der Waals\s*(\-?\d*\.\d*)",
        r"Atomic Multipoles\s*(\-?\d*\.\d*)",
        r"Polarization\s*(\-?\d*\.\d*)",
        r"NN Valence\s*(\-?\d*\.\d*)",
        r"NN Metal\s*(\-?\d*\.\d*)",
        r"NN Term\s*(\-?\d*\.\d*)",
        r"Geometric Restraints\s*(\-?\d*\.\d*)",
        r"Urey-Bradley\s*(\-?\d*\.\d*)"
    ]

    def __init__(
        self,
        final_xyz=None,
        final_xyz_text=None,
        final_key=None,
        final_key_text=None,
        scratch_dir="/scratch/yw24267/",
        tinker_path="/home/yw24267/programs/tinker/Tinker8/latest/bin/",
        num_threads=1,
        timeout=300,
    ) -> None:
        assert final_xyz or final_xyz_text, "final_xyz or final_xyz_text must be provided."
        assert not final_xyz or not final_xyz_text, "final_xyz and final_xyz_text cannot be provided at the same time."
        assert final_key or final_key_text, "final_key or final_key_text must be provided."
        assert not final_key or not final_key_text, "final_key and final_key_text cannot be provided at the same time."
        if final_xyz:
            self.final_xyz = os.path.abspath(final_xyz)
            with open(self.final_xyz) as f:
                self.final_xyz_text = f.read().strip().split("\n")
        else:
            self.final_xyz_text = final_xyz_text.strip().split("\n")
        self.scratch_dir = scratch_dir
        self.tinker_path = tinker_path
        self.timeout = timeout
        self.num_threads = num_threads
        self.final_key_text = final_key_text
        self.final_key = final_key
        if self.final_key:
            self.final_key = os.path.abspath(self.final_key)
            assert os.path.isfile(self.final_key), f"Key File Does Not Exist! {self.final_key}"

        self.num_headerlines = 1 + TinkerXYZ._is_box_info(self.final_xyz_text[1])
        os.makedirs(self.scratch_dir, exist_ok=True)

    def write_txyz(self, xyz=None, save_path=None, verbose_rtn=False):
        """write a temporary xyz file for tinker to read, if xyz is None, use the final_xyz file."""
        xyz_block = list(self.final_xyz_text)  # make an independent copy

        # initilization of variables
        dim_expansion_flag = False
        num_of_confs = 1

        if xyz is not None:
            xyz = np.array(xyz)
            if xyz.ndim == 2:
                dim_expansion_flag = True
                xyz = np.expand_dims(xyz, axis=0)
            assert xyz.ndim == 3, f"shape disallowed for xyz {xyz.shape}"
            assert len(xyz_block) == xyz.shape[1] + self.num_headerlines, f"number of atoms mismatch {len(xyz_block) - 1} vs. {xyz.shape[1]}. Input files: {self.final_key} {self.final_xyz} {self.final_xyz_text}."
            num_of_confs = xyz.shape[0]
            xyz_block = [i.split() for i in xyz_block]
            xyz_string = ""
            for xyz_i in list(xyz):
                xyz_i_string = list(xyz_block)
                for line, pos in zip(xyz_i_string[self.num_headerlines:], xyz_i):
                    line[2:5] = map(str, pos)
                xyz_i_string = "\n".join(["\t".join(i) for i in xyz_i_string])
                # previously used two line breaks (ie one blank line) to separate frames, which works with tinker8 
                # but tinker9 does not like it, giving only correct results for the 1st frame.
                # xyz_string += xyz_i_string + "\n\n"
                xyz_string += xyz_i_string + "\n"
        else:
            xyz_string = "\n".join(self.final_xyz_text)

        if save_path:
            tmp_xyz_file = save_path
        else:
            tmp_folder = f"{self.scratch_dir}/amoebann.ts{time.time()}.pid{os.getpid()}"
            os.makedirs(tmp_folder)
            tmp_xyz_file = f"{tmp_folder}/tmp.xyz"
        with open(tmp_xyz_file, "w") as f:
            f.write(xyz_string)

        if verbose_rtn:
            return tmp_xyz_file, dim_expansion_flag, num_of_confs
        else:
            return tmp_xyz_file
        
    def write_key(self, tmp_xyz_file):
        if self.final_key:
            return self.final_key
        else:
            assert self.final_key_text, "final_key_text must be provided."
            tmp_key_file = tmp_xyz_file[:-4] + ".key"
            with open(tmp_key_file, "w") as f:
                f.write(self.final_key_text)
            return tmp_key_file
        
    def prepare4nn(self, nnkey, tinker, cuda):
        if os.path.isfile(nnkey):
            os.system(f"cp {nnkey} {tinker.wd}/tinker9nn.yaml")
        else:
            with open(f"{tinker.wd}/tinker9nn.yaml", "w") as f:
                f.write(nnkey)
        ambnn_path = os.path.join(os.path.dirname(tinker.tinker_path.rstrip('/')), "ext")
        pre_cmds = (
            f"source /work/yw24267/miniconda3/etc/profile.d/conda.sh && conda activate cuda11"
            f" && export PYTHONPATH={ambnn_path}:$PYTHONPATH"
            f" && export CUDA_VISIBLE_DEVICES={cuda}"
        )
        return pre_cmds

    def get_energy(self, xyz=None, nnkey=None, cuda=0):
        # TODO check if the total ene == the sum of components
        """replace the xyz coordinates and calculate energy

        Args:
            xyz (2d / 3d array, optional): array of xyz coordinates, np.ndarray / nested list. Defaults to None.

        Returns:
            tuple of floats: total energy and decompositions
        """
        tmp_xyz_file, dim_expansion_flag, num_of_confs = self.write_txyz(xyz=xyz, verbose_rtn=True)
        tmp_key_file = self.write_key(tmp_xyz_file)

        try:
            tinker = Tinker(wd=os.path.dirname(tmp_xyz_file), tinker_path=self.tinker_path, timeout=self.timeout)
            if nnkey:
                pre_cmds = self.prepare4nn(nnkey, tinker, cuda)
                outputs, _ = tinker.call("analyze9", cmd_args=f"{tmp_xyz_file} -k {tmp_key_file} E", pre_cmds=pre_cmds)
            elif os.path.isfile(f"{self.tinker_path}/analyze9"):
                outputs, _ = tinker.call("analyze9", cmd_args=f"{tmp_xyz_file} -k {tmp_key_file} E")
            else:
                # -t overwrites threads assigned in the key file.
                outputs, _ = tinker.call("analyze", cmd_args=f"{tmp_xyz_file} -t {self.num_threads} -k {tmp_key_file} E")
            # print(outputs)
            outputs = outputs.split("Analysis for Archive Structure :")  # split out energies for each conf 
            energies = []
            for outputs_i in outputs:
                energies_i = []
                for p in self.energy_patts:
                    e = re.findall(p, outputs_i)
                    e = float(e[0]) if e else 0
                    energies_i.append(e)
                energies.append(energies_i)
        except subprocess.CalledProcessError:
            energies = [["ERROR",] * len(self.energy_patts)] * num_of_confs
        except subprocess.TimeoutExpired:
            energies = [["TIMEOUT",] * len(self.energy_patts)] * num_of_confs

        # os.remove(tmp_xyz_file)
        shutil.rmtree(tinker.wd)
        if dim_expansion_flag:
            energies = energies[0]
        return energies
    
    def get_gradients(self, xyz=None, nnkey=None, cuda=0, num_digits=8):
        """replace the xyz coordinates and calculate gradients

        Args:
            xyz (2d / 3d array, optional): array of xyz coordinates, np.ndarray / nested list. Defaults to None.
            nnkey (str, optional): path to the nnkey file. Must be a yaml file or None. Defaults to None.

        Returns:
            tuple of tuple of floats: gradients
        """
        tmp_xyz_file, dim_expansion_flag, num_of_confs = self.write_txyz(xyz=xyz, verbose_rtn=True)
        tmp_key_file = self.write_key(tmp_xyz_file)

        try:
            tinker = Tinker(wd=os.path.dirname(tmp_xyz_file), tinker_path=self.tinker_path, timeout=self.timeout)
            if nnkey:
                pre_cmds = self.prepare4nn(nnkey, tinker, cuda)
                outputs, _ = tinker.call("testgrad9", cmd_args=f"{tmp_xyz_file} -k {tmp_key_file}", pre_cmds=pre_cmds)
            elif os.path.isfile(f"{self.tinker_path}/testgrad9"):
                outputs, _ = tinker.call("testgrad9", cmd_args=f"{tmp_xyz_file} -k {tmp_key_file}")
            else:
                outputs, _ = tinker.call("testgrad", cmd_args=f"{tmp_xyz_file} -t {self.num_threads} -k {tmp_key_file} Y N N")
            outputs = outputs.split("Analysis for Archive Structure :")  # split out grads for each conf 
            grads = []
            for outputs_i in outputs:
                patt = r"Anlyt\s+(\d+)\s*((?:\s|\-|\d){1,7}\.\d{1,@nd})((?:\s|\-|\d){1,7}\.\d{1,@nd})((?:\s|\-|\d){1,7}\.\d{1,@nd})\s*\d*\.\d*"
                patt = patt.replace("@nd", str(num_digits))
                grads_i = re.findall(patt, outputs_i)
                grads_i = sorted(grads_i, key=lambda x: int(x[0]))
                grads_i = [(float(gx), float(gy), float(gz)) for _, gx, gy, gz in grads_i]
                grads.append(grads_i)
        except subprocess.CalledProcessError:
            grads = ["ERROR"] * num_of_confs
        except subprocess.TimeoutExpired:
            grads = ["TIMEOUT"] * num_of_confs

        shutil.rmtree(tinker.wd)
        if dim_expansion_flag:
            grads = grads[0]
        return grads


class NeuralNetworkPythonBackend:
    # config, load model
    def __init__(self) -> None:
        self._initilized = False

    def initialize(self):
        if self._initilized:
            return
        self._config = load_yaml(os.path.join(os.getcwd(), "tinker9nn.yaml"))

        # nn_model & device
        ckpt = self.config.get("nn_model", '')
        assert os.path.isfile(ckpt), f"Cannot find the model file. Please assign a valid `nn_model` parameter."
        dev = self.config.get("device", "cuda")
        self._model = get_ckpt_model(model_ckpt=ckpt, device=dev).to(torch.double)
        # nn_atoms
        # TODO assure all groups do not overlap. they should be different molecules by design.
        if "nn_atoms" in self.config:
            for idx, grp in enumerate(self.config["nn_atoms"]):
                grp_ = []
                for i in grp:
                    if isinstance(i, int):
                        # the input index starts at 1. move the index to 0-based
                        grp_.append(i-1)
                    elif isinstance(i, list):
                        i[0] -= 1  # so that it's inclusive at both ends
                        grp_.extend(list(range(*i)))
                    else:
                        raise TypeError(f"Invalid {i} in {grp} for nn_atoms.")
                self.config["nn_atoms"][idx] = grp_

        self._initilized = True

    @staticmethod
    def _get_ckpt_model(model_ckpt, device="cuda"):
        """deprecated, kept just for compatibility"""
        model = get_ckpt_model(model_ckpt=model_ckpt, device=device).to(torch.double)
        return model

    @property
    def model(self):
        return self._model
    
    @property
    def config(self):
        return self._config


nnbkd = NeuralNetworkPythonBackend()


def nn_analyze(atm_nums, xyzs, is_bonded):
    """get energy and gradient from nn of one molecule
    atm_nums: list of atomic numbers
    xyzs: list of atomic coordinates
    """
    # torch.use_deterministic_algorithms(True)
    nnbkd.initialize()
    # print(nnbkd.config)

    # start = time.time()
    nn_ene = 0
    xyzs = torch.tensor(xyzs, dtype=torch.double)
    nn_grads = torch.zeros_like(xyzs)
    if is_bonded == nnbkd.config.get("is_bonded", True):
        nn_atoms = nnbkd.config.get("nn_atoms", [])
        if not nn_atoms:
            nn_atoms = [list(range(len(atm_nums)))]
            nnbkd.config["nn_atoms"] = nn_atoms
        
        atomic2symbol = {1: "H", 6: "C", 7: "N", 8: "O", 29: "Cu"}
        # {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"}
        speices = torch.tensor(ANINetworkDataset.symbol2index([
            atomic2symbol[i] for i in atm_nums
        ]))

        # print("nn_atoms:", nn_atoms, flush=True)
        # print('config["supported_species"]:', config["supported_species"], flush=True)
        nn_atoms = [[i for i in grp if atomic2symbol[atm_nums[i]] in config["supported_species"]] for grp in nn_atoms]
        # print("nn_atoms:", nn_atoms, flush=True)

        for batched_grps in zip_longest(*[iter(nn_atoms),] * 100):
            batched_grps = [grp for grp in batched_grps if grp is not None]
            data = ANINetworkDataset.collate_batch([
                {
                    "species": speices[grp], 
                    "coordinates": xyzs[grp],
                } for grp in batched_grps
            ])

            nn_enes_raw, nn_grads_raw = nnbkd.model.analyze(data)
            # nn_enes_raw *= 0
            # nn_grads_raw *= 0
            nn_ene += nn_enes_raw.sum()
            for idx, grp in enumerate(batched_grps):
                nn_grads[grp] = nn_grads_raw[idx][:len(grp)]

        if nnbkd.config.get("nn_atoms_xyzs", []):
            # substract amoeba energy and forces for the group
            xyz_files = nnbkd.config.get("nn_atoms_xyzs", [])
            key_files = nnbkd.config.get("nn_atoms_keys", [])
            for xyz_file, key_file, grp in zip(xyz_files, key_files, nn_atoms):
                amb = AMOEBACalculator(final_xyz=xyz_file, final_key=key_file)
                grads = amb.get_gradients(xyzs[grp].tolist())
                nn_grads[grp] -= torch.tensor(grads).to(torch.double)
                nn_ene -= amb.get_energy(xyzs[grp].tolist())[0]
                
        nn_ene = nn_ene.item()

    # print("Max Gradients:", torch.max(nn_grads).item(), flush=True, end=" ")
    # print("Min Gradients:", torch.min(nn_grads).item(), flush=True)
    # nn_grads = (nn_grads / 2).tolist()
    nn_grads_x, nn_grads_y, nn_grads_z = list(zip(*nn_grads.tolist()))
    # print("Py:", xyzs[0], nn_enes, nn_grads_x[0], nn_grads_y[0], nn_grads_z[0], flush=True)
    # print("enes, grads:", type(nn_enes), type(nn_grads_x[0]), type(nn_grads_y[0]), type(nn_grads_z[0]), flush=True)
    # print("nn_ene:", nn_ene, flush=True)
    # print("nn_grads:\n", nn_grads, flush=True)
    # raise KeyboardInterrupt("just for debug")
    # exit()

    # stop = time.time()
    # print(f"nnvalence py time: {stop - start}", flush=True)
    return nn_ene, nn_grads_x, nn_grads_y, nn_grads_z