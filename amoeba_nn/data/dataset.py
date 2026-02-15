# -*- coding: utf-8 -*-
# @Time       : 2022/05/11 14:29:29
# @Author     : Yanxing Wang <yxw@utexas.edu>
# @Project    : amoeba_nn
# @Description: dataset classes for different sources of data

import logging
import re
import time
import os
import sys
import shutil
import subprocess
import pickle
from itertools import chain
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import tables
import h5py
import torch
from torch.utils.data import DataLoader
from rdkit import Chem

from ..third_party.pyanitools import anidataloader
from ..utils.config import config
from ..utils.helpers import classproperty


logger = logging.getLogger(__name__)


class ANIDatabaseH5:
    """class for convient indexing in ANI HDF5 files.
    """

    def __init__(self, h5_files=None, h5_cache=False, h5_inmemory=False):
        self._h5_files = dict(zip([re.findall(r"(s\d\d)\.", i)[0] for i in h5_files], h5_files))
        self.h5_cache = h5_cache
        self.h5_inmemory = h5_inmemory
        if self.h5_cache:
            self._h5_file_objects = {k: tables.open_file(v) for k, v in self._h5_files.items()}
        if self.h5_inmemory:
            self._h5_file_objects = {k: tables.open_file(v, driver="H5FD_CORE") for k, v in self._h5_files.items()}

    def __getitem__(self, mol_id):
        """indexing of ani data

        Args:
            mol_id (str or [str, int] or [str, slice]): mol_id (e.g., "gdb11_s01-0") alone, or with numeric index / slice of conformers. 

        Returns:
            dict: dictionary with three keys.
        """
        if len(mol_id) == 2:
            mol_id, conf_id = mol_id
        else:
            conf_id = slice(None)
        subset = re.findall(r"(s\d\d)\-", mol_id)[0]
        data = {}
        # keys_1 = ['smiles', 'species']
        keys_1 = ["species"]
        keys_2 = ["coordinates", "energies"]
        if self.h5_cache or self.h5_inmemory:
            f = self._h5_file_objects[subset]
        else:
            f = tables.open_file(self._h5_files[subset])
        for k in keys_1:
            data[k] = f.root[f"gdb11_{subset}"][mol_id][k][:]
        for k in keys_2:
            data[k] = f.root[f"gdb11_{subset}"][mol_id][k][conf_id]
        if not (self.h5_cache or self.h5_inmemory):
            f.close()
        # data['smiles'] = ''.join([i.decode('ascii') for i in data['smiles']])
        data["species"] = [i.decode("ascii") for i in data["species"]]
        return data

    def __iter__(self):
        h5_files = sorted(self._h5_files.values())
        return chain(*[anidataloader(h5) for h5 in h5_files])

    def __del__(self):
        if self.h5_cache or self.h5_inmemory:
            for f in self._h5_file_objects.values():
                f.close()


class SPICEDatabaseH5:
    """class for convient indexing in SPICE HDF5 files.
    """
    def __init__(self, h5_file=None, h5_cache=False, h5_inmemory=False, ANI1_like=False):
        self._h5_file = h5_file
        self.h5_cache = h5_cache
        self.h5_inmemory = h5_inmemory
        self.ANI1_like = ANI1_like
        if self.h5_cache:
            self._h5_file_object = tables.open_file(self._h5_file)
        if self.h5_inmemory:
            self._h5_file_object = tables.open_file(self._h5_file, driver="H5FD_CORE")

    def __getitem__(self, mol_id):
        """indexing of spice data

        Args:
            mol_id (str or [str, int] or [str, slice]): mol_id (e.g., "ala-ala") alone, or with numeric index / slice of conformers. 

        Returns:
            dict: dictionary with three keys.
        """
        if len(mol_id) == 2:
            mol_id, conf_id = mol_id
        else:
            conf_id = slice(None)
        data = {}
        keys_1 = ["atomic_numbers"]
        keys_2 = ["conformations", "dft_total_energy", "dft_total_gradient"]
        
        if self.h5_cache or self.h5_inmemory:
            f = self._h5_file_object
        else:
            f = tables.open_file(self._h5_file)

        for k in keys_1:
            data[k] = f.root[mol_id][k][:]
        for k in keys_2:
            data[k] = f.root[mol_id][k][conf_id]
        # data['smiles'] = f.root[mol_id]['smiles'].decode()  # tables does not support the datatype used for smiles

        if not (self.h5_cache or self.h5_inmemory):
            f.close()
        if self.ANI1_like:
            data = self._make_it_ANI1_like(data)
        return data

    @staticmethod
    def _make_it_ANI1_like(data):
        key_mapping = {
            "atomic_numbers": "species",
            "conformations": "coordinates",
            "dft_total_energy": "energies",
        }
        data_ = {key_mapping.get(k, k): v for k, v in data.items()}
        data_["species"] = [Chem.Atom(int(i)).GetSymbol() for i in data_["species"]]
        bohr2angtrom = 0.529177249
        data_["coordinates"] *= bohr2angtrom
        return data_

    def __iter__(self):
        h5 = h5py.File(self._h5_file)
        for g_id, g_data in h5.items():
            subset = g_data["subset"][:][0].decode()
            cols = [
                "atomic_numbers",
                "conformations",
                "dft_total_energy",
                "dft_total_gradient",
                # "formation_energy",
                # "smiles",
                # "subset",
            ]
            yield {
                g_id: {
                    **{
                        "smiles": g_data["smiles"][:][0].decode(),
                        "subset": subset,
                    },
                    **{c: g_data[c][:] for c in cols},
                }
            }

    def __del__(self):
        if self.h5_cache or self.h5_inmemory:
            self._h5_file_object.close()


class ANI1xDatabaseH5:
    """class for convient indexing in ANI1x HDF5 files.
    """
    def __init__(self, h5_file=None, h5_cache=False, h5_inmemory=False, ANI1_like=False):
        self._h5_file = h5_file
        self.h5_cache = h5_cache
        self.h5_inmemory = h5_inmemory
        self.ANI1_like = ANI1_like
        if self.h5_cache:
            self._h5_file_object = tables.open_file(self._h5_file)
        if self.h5_inmemory:
            self._h5_file_object = tables.open_file(self._h5_file, driver="H5FD_CORE")

    def __getitem__(self, mol_id):
        """indexing of spice data

        Args:
            mol_id (str or [str, int] or [str, slice]): mol_id alone, or with numeric index of conformers. 

        Returns:
            dict: dictionary with three keys.
        """
        if len(mol_id) == 2:
            mol_id, conf_id = mol_id
            if isinstance(conf_id, slice):
                NotImplementedError(
                    "Indexing without a conf_id is not supported, "
                    "because the conformation index in ANIx is not consecutive."
                    "Slicing may cause errors!"
                )
        else:
            # raise NotImplementedError(
            #     "Indexing without a conf_id is not supported, "
            #     "because the conformation index in ANIx is not consecutive and thus not having a default range."
            # )
            # conf_id = slice(None)
            conf_id = int(mol_id.split('-')[-1])
        mol_id = mol_id.split('-')[0]
        data = {}
        keys_1 = ["atomic_numbers"]
        keys_2 = ["coordinates", "wb97x_dz.energy", "wb97x_dz.forces", 'wb97x_tz.energy', 'wb97x_tz.forces']
        
        if self.h5_cache or self.h5_inmemory:
            f = self._h5_file_object
        else:
            f = tables.open_file(self._h5_file)

        for k in keys_1:
            data[k] = f.root[mol_id][k][:]
        for k in keys_2:
            data[k] = f.root[mol_id][k][conf_id]
        # data['smiles'] = f.root[mol_id]['smiles'].decode()  # tables does not support the datatype used for smiles

        if not (self.h5_cache or self.h5_inmemory):
            f.close()
        if self.ANI1_like:
            data = self._make_it_ANI1_like(data)
        return data

    @staticmethod
    def _make_it_ANI1_like(data):
        key_mapping = {
            "atomic_numbers": "species",
            "wb97x_dz.energy": "energies",
            "wb97x_dz.forces": "forces",
        }
        data_ = {key_mapping.get(k, k): v for k, v in data.items()}
        data_["species"] = [Chem.Atom(int(i)).GetSymbol() for i in data_["species"]]
        return data_

    def __iter__(self):
        for data in self._iter_data_buckets():
            yield {
                data["name"]: data,
            }

    def _iter_data_buckets(self, keys=['wb97x_dz.energy']):
        """ Iterate over buckets of data in ANI HDF5 file. 
        Yields dicts with atomic numbers (shape [Na,]) coordinated (shape [Nc, Na, 3])
        and other available properties specified by `keys` list, w/o NaN values.
        """
        keys = set(keys)
        keys.discard('atomic_numbers')
        keys.discard('coordinates')
        with h5py.File(self._h5_file, 'r') as f:
            for grp in f.values():
                Nc = grp['coordinates'].shape[0]
                mask = np.ones(Nc, dtype=np.bool)
                data = dict((k, grp[k][()]) for k in keys)
                for k in keys:
                    v = data[k].reshape(Nc, -1)
                    mask = mask & ~np.isnan(v).any(axis=1)
                if not np.sum(mask):
                    continue
                d = dict((k, data[k][mask]) for k in keys)
                d['name'] = os.path.basename(grp.name)
                d['atomic_numbers'] = grp['atomic_numbers'][()]
                d['coordinates'] = grp['coordinates'][()][mask]
                yield d 

    def __del__(self):
        if self.h5_cache or self.h5_inmemory:
            self._h5_file_object.close()

    def get_xyz_block(self, mol_id, conf_id=None):
        data = self[mol_id, conf_id] if conf_id else self[mol_id]
        data = data if self.ANI1_like else self._make_it_ANI1_like(data) 
        sym = data["species"]
        pos = data["coordinates"]
        xyz = f"{len(sym)}\n\n"
        for s, p in zip(sym, pos.tolist()):
            p = list(map(str, p))
            p.insert(0, s)
            xyz += "\t".join(p)+"\n"
        return xyz


class ANI1xDatabaseH5Clean(ANI1xDatabaseH5):
    """class for convient indexing in the ANI1x HDF5 file that has been processed to reorder the atoms.
    """
    def __init__(self, h5_file=None, h5_cache=False, h5_inmemory=False, ANI1_like=False):
        super().__init__(h5_file, h5_cache, h5_inmemory, ANI1_like)

    def __getitem__(self, mol_id):
        """indexing of spice data

        Args:
            mol_id (str or [str, int] or [str, slice]): mol_id (e.g., "ala-ala") alone, or with numeric index / slice of conformers. 

        Returns:
            dict: dictionary with three keys.
        """
        if len(mol_id) == 2:
            mol_id, conf_id = mol_id
        else:
            conf_id = slice(None)
        data = {}
        keys_1 = ["atomic_numbers"]
        keys_2 = ["coordinates", "wb97x_dz.energy", "wb97x_dz.forces", 'wb97x_tz.energy', 'wb97x_tz.forces', 
                  'orig_conf_ids', 'orig_atom_orders']
        # keys_2 = ["coordinates", "wb97x_dz.energy"]
        
        if self.h5_cache or self.h5_inmemory:
            f = self._h5_file_object
        else:
            f = tables.open_file(self._h5_file)

        for k in keys_1:
            data[k] = f.root[mol_id][k][:]
        for k in keys_2:
            data[k] = f.root[mol_id][k][conf_id]
        # data['smiles'] = f.root[mol_id]['smiles'].decode()  # tables does not support the datatype used for smiles

        if not (self.h5_cache or self.h5_inmemory):
            f.close()
        if self.ANI1_like:
            data = self._make_it_ANI1_like(data)
        return data

    def __iter__(self):
        h5 = h5py.File(self._h5_file)
        for g_id, g_data in h5.items():
            cols = [
                "atomic_numbers",
                "coordinates", 
                "wb97x_dz.energy", 
                # "wb97x_dz.forces", 
                # 'wb97x_tz.energy', 
                # 'wb97x_tz.forces',
            ]
            yield {
                g_id: {
                    **{c: g_data[c][:] for c in cols},
                }
            }


class MetalDatabaseH5(ANI1xDatabaseH5):
    """class for convient indexing in the metal HDF5 file.
    """
    def __init__(self, h5_file=None, h5_cache=False, h5_inmemory=False, ANI1_like=False):
        super().__init__(h5_file, h5_cache, h5_inmemory, ANI1_like)

    def __getitem__(self, mol_id):
        """indexing of spice data

        Args:
            mol_id (str or [str, int] or [str, slice]): mol_id (e.g., "ala-ala") alone, or with numeric index / slice of conformers. 

        Returns:
            dict: dictionary with three keys.
        """
        if len(mol_id) == 2:
            mol_id, conf_id = mol_id
        else:
            conf_id = slice(None)
        data = {}
        keys_1 = ["atomic_numbers"]
        keys_2 = ["coordinates", "wb97x.energy", "wb97x.force", "amoeba.energy", "amoeba.force"]
        
        if self.h5_cache or self.h5_inmemory:
            f = self._h5_file_object
        else:
            f = tables.open_file(self._h5_file)

        for k in keys_1:
            data[k] = f.root[mol_id][k][0]
        for k in keys_2:
            data[k] = f.root[mol_id][k][conf_id]
        # data['smiles'] = f.root[mol_id]['smiles'].decode()  # tables does not support the datatype used for smiles

        if not (self.h5_cache or self.h5_inmemory):
            f.close()
        if self.ANI1_like:
            data = self._make_it_ANI1_like(data)
        return data
    
    @staticmethod
    def _make_it_ANI1_like(data):
        key_mapping = {
            "atomic_numbers": "species",
            "wb97x.energy": "energies",
            "wb97x.force": "forces",
        }
        data_ = {key_mapping.get(k, k): v for k, v in data.items()}
        data_["species"] = [Chem.Atom(int(i)).GetSymbol() for i in data_["species"]]
        return data_

    def __iter__(self):
        h5 = h5py.File(self._h5_file)
        for g_id, g_data in h5.items():
            cols = [
                "atomic_numbers",
                "coordinates", 
                "wb97x.energy", 
                "wb97x.force", 
                "amoeba.energy", 
                "amoeba.force",
            ]
            yield {
                g_id: {
                    **{c: g_data[c][:] for c in cols},
                }
            }


class TetraPeptDatabaseH5(ANI1xDatabaseH5):
    """class for convient indexing in the HDF5 file of tetra peptide data from Dr Yong Duan.
    """
    def __init__(self, h5_file=None, h5_cache=False, h5_inmemory=False, ANI1_like=False):
        super().__init__(h5_file, h5_cache, h5_inmemory, ANI1_like)

    def __getitem__(self, mol_id):
        """indexing of the data

        Args:
            mol_id (str or [str, int] or [str, slice]): mol_id (e.g., "ala-ala") alone, or with numeric index / slice of conformers. 

        Returns:
            dict: dictionary with three keys.
        """
        if len(mol_id) == 2:
            mol_id, conf_id = mol_id
        else:
            conf_id = slice(None)
        data = {}
        keys_1 = ["species"]
        keys_2 = ["coordinates"]
        
        if self.h5_cache or self.h5_inmemory:
            f = self._h5_file_object
        else:
            f = tables.open_file(self._h5_file)

        for k in keys_1:
            data[k] = f.root[mol_id][k][:]
        for k in keys_2:
            data[k] = f.root[mol_id][k][conf_id]
        data["species"] = [i.decode("ascii") for i in data["species"]]

        if not (self.h5_cache or self.h5_inmemory):
            f.close()

        return data

    def __iter__(self):
        h5 = h5py.File(self._h5_file)
        for g_id, g_data in h5.items():
            cols = [
                "species",
                "coordinates", 
            ]
            yield {
                g_id: {
                    **{c: g_data[c][:] for c in cols},
                }
            }


class MultipoleH5:
    """class for convient indexing in multipole HDF5 files.
    """

    def __init__(self, h5_file=None):
        self._h5_file_object = tables.open_file(h5_file, driver="H5FD_CORE")
        self.keys = ["AMOBEA_AtomTypes", "AMOBEA_AtomMultipoles"]

    def __getitem__(self, mol_id):
        """indexing of data

        Args:
            mol_id (str): mol_id (e.g., "gdb11_s01-0") 

        Returns:
            dict: dictionary with two keys.
        """
        data = {}
        for k in self.keys:
            data[k] = self._h5_file_object.root[mol_id][k][:]
        return data

    def __iter__(self):
        for i in self._h5_file_object.iter_nodes("/"):
            # try:
            #     data = {k: i[k][:] for k in self.keys}
            # except:
            #     data = {}
            # finally:
            #     data["Name"] = i._v_name
            #     yield data
            data = {k: i[k][:] for k in self.keys}
            data["Name"] = i._v_name
            yield data

    def __del__(self):
        self._h5_file_object.close()


class ForceH5:
    """class for convient indexing in force HDF5 files.
    """

    def __init__(self, h5_file=None):
        self._h5_file_object = tables.open_file(h5_file, driver="H5FD_CORE")
        self.keys = ["forces"]

    def __getitem__(self, mol_id):
        """indexing of data

        Args:
            mol_id (str): mol_id (e.g., "gdb11_s01-0") 

        Returns:
            dict: dictionary with two keys.
        """
        if len(mol_id) == 2:
            mol_id, conf_id = mol_id
        else:
            conf_id = slice(None)

        data = {}
        for k in self.keys:
            data[k] = self._h5_file_object.root[mol_id][k][conf_id]
        return data

    # def __iter__(self):
    #     for i in self._h5_file_object.iter_nodes("/"):
    #         # try:
    #         #     data = {k: i[k][:] for k in self.keys}
    #         # except:
    #         #     data = {}
    #         # finally:
    #         #     data["Name"] = i._v_name
    #         #     yield data
    #         data = {k: i[k][:] for k in self.keys}
    #         data["Name"] = i._v_name
    #         yield data

    def __del__(self):
        self._h5_file_object.close()


class ANINetworkDataset:
    """Dataset for ANI Network
    """

    def __init__(
        self, df, h5_files, h5_multipole=None, h5_force=None, label_col=None, samp_w_col=None, shuffle=False, h5_cache=False, h5_inmemory=False, device="cpu"
    ) -> None:
        # load data for ani network
        self._prepare_df(df, label_col, samp_w_col, shuffle)
        self.mp = self._prepare_other_h5(MultipoleH5, h5_multipole)
        self.force = self._prepare_other_h5(ForceH5, h5_force)
        self.device = device
        self.set_h5db(h5_files, h5_cache, h5_inmemory)
    
    def set_h5db(self, h5_files, h5_cache, h5_inmemory):
        # this function needs to be overrided by child class
        self.h5db = ANIDatabaseH5(h5_files=h5_files, h5_cache=h5_cache, h5_inmemory=h5_inmemory)
    
    def _prepare_df(self, df, label_col, samp_w_col, shuffle):
        self.df = df.reset_index(drop=True)
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        # print(label_col, self.df.columns)
        self.label_col = label_col if label_col else [c for c in self.df.columns if c[:5] == "LABEL"][0]
        assert self.label_col in self.df.columns, "label_col is invaild"
        if samp_w_col:
            assert self.samp_w_col in self.df.columns, "samp_w_col is invaild"
            self.samp_w_col = samp_w_col
        else:
            self.samp_w_col = "SAmP_W_DeFalt"
            self.df[self.samp_w_col] = 1

    def _prepare_other_h5(self, h5_class, h5_file):
        if h5_file:
            return h5_class(h5_file)
        else:
            return None

    def __getitem__(self, index):
        # support (ID, CONF_ID) and numeric indices.
        logger.debug(f"start to fetch table data for index {index}")
        if type(index) in (int, torch.int, torch.long):
            index = len(self) + index if index < 0 else index
            mol_id, conf_id, energy, samp_w = self.df.loc[index, ["ID", "CONF_ID", self.label_col, self.samp_w_col]]
        elif len(index) == 2:
            mol_id, conf_id = index
            energy, samp_w = self.df.query("ID == @mol_id & CONF_ID == @conf_id").iloc[0, :][[self.label_col, self.samp_w_col]]
        else:
            raise NotImplementedError(f"Unable to parse {index} as valid index")

        logger.debug(f"start to fetch xyz data for index {index}")
        data = self.h5db[mol_id, conf_id]
        rtn = {
            "species": torch.tensor(self.symbol2index(data["species"]), dtype=torch.long, device=self.device),
            "coordinates": torch.tensor(data["coordinates"], dtype=torch.float, device=self.device),
            "energies": torch.tensor(energy, dtype=torch.float, device=self.device),
            "sample_weights": torch.tensor(samp_w, dtype=torch.float, device=self.device),
        }
        if self.force is not None:
            data_f = self.force[mol_id, conf_id]
            rtn["forces"] = torch.tensor(data_f["forces"], dtype=torch.float, device=self.device)
        if self.mp is not None:
            data_mp = self.mp[mol_id]
            rtn["multipoles"] = torch.tensor(data_mp["AMOBEA_AtomMultipoles"], dtype=torch.float, device=self.device)
        logger.debug(f"all data fetched for index {index}")
        return rtn

    def __repr__(self):
        return str(self.df)

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @classproperty
    def supported_species(cls):
        """supported species for the dataset
        """
        return config["supported_species2idx"]

    @classmethod
    def symbol2index(cls, symbols):
        if isinstance(symbols, str):
            symbols = [symbols]
        symbols = [cls.supported_species.get(i, -1) for i in symbols]
        return symbols
    
    @classmethod
    def atmnum2index(cls, atomic_numbers):
        if isinstance(atomic_numbers, int):
            atomic_numbers = [atomic_numbers]
        symbols = [Chem.Atom(i).GetSymbol() for i in atomic_numbers]
        symbols = [cls.supported_species.get(i, -1) for i in symbols]
        return symbols

    padding_values = {
        "species": -1,
        "coordinates": 0,
        "multipoles": 0,
        "forces": 0,
    }

    @classmethod
    def collate_batch(cls, batch):
        logger.debug("start to collate batch")
        output = {k: [] for k in batch[0].keys()}
        for i in batch:
            for k, v in i.items():
                output[k].append(v)
        for k, v in output.items():
            if v[0].dim() == 0:
                output[k] = torch.stack(v)
            else:
                output[k] = torch.nn.utils.rnn.pad_sequence(v, True, cls.padding_values[k])
        logger.debug("end of collate batch")
        return output


class ANINetworkDatasetForMetal(ANINetworkDataset):
    """Dataset class working with metal dataset for ANI Network
    """
    def __init__(
        self, df, h5_files, h5_multipole=None, h5_force=None, label_col=None, samp_w_col=None, shuffle=False, h5_cache=False, h5_inmemory=False, device="cpu"
    ) -> None:
        super().__init__(df, h5_files, h5_multipole=h5_multipole, h5_force=h5_force,
                         label_col=label_col, samp_w_col=samp_w_col, shuffle=shuffle, h5_cache=h5_cache, h5_inmemory=h5_inmemory, device=device)
        
    def set_h5db(self, h5_files, h5_cache, h5_inmemory):
        self.h5db = MetalDatabaseH5(h5_file=h5_files[0], h5_cache=h5_cache, h5_inmemory=h5_inmemory, ANI1_like=True)


class ANINetworkDatasetForSPICE(ANINetworkDataset):
    """Dataset class working with SPICE for ANI Network
    """
    def __init__(
        self, df, h5_files, h5_multipole=None, h5_force=None, label_col=None, samp_w_col=None, shuffle=False, h5_cache=False, h5_inmemory=False, device="cpu"
    ) -> None:
        super().__init__(df, h5_files, h5_multipole, 
                         label_col=label_col, samp_w_col=samp_w_col, shuffle=shuffle, h5_cache=h5_cache, h5_inmemory=h5_inmemory, device=device)
        # # load data for ani network
        # self._prepare_df(df, label_col, samp_w_col, shuffle)
        # self._prepare_multipole_h5(h5_multipole)
        # self.device = device
        
    def set_h5db(self, h5_files, h5_cache, h5_inmemory):
        self.h5db = SPICEDatabaseH5(h5_file=h5_files[0], h5_cache=h5_cache, h5_inmemory=h5_inmemory, ANI1_like=True)


class ANINetworkDatasetForANI1x(ANINetworkDataset):
    """Dataset class working with ANI1x for ANI Network
    """
    def __init__(
        self, df, h5_files, h5_multipole=None, h5_force=None, label_col=None, samp_w_col=None, shuffle=False, h5_cache=False, h5_inmemory=False, device="cpu"
    ) -> None:
        super().__init__(df, h5_files, h5_multipole=h5_multipole, h5_force=h5_force,
                         label_col=label_col, samp_w_col=samp_w_col, shuffle=shuffle, h5_cache=h5_cache, h5_inmemory=h5_inmemory, device=device)
        # # load data for ani network
        # self._prepare_df(df, label_col, samp_w_col, shuffle)
        # self._prepare_multipole_h5(h5_multipole)
        # self.device = device
        
    def set_h5db(self, h5_files, h5_cache, h5_inmemory):
        self.h5db = ANI1xDatabaseH5Clean(h5_file=h5_files[0], h5_cache=h5_cache, h5_inmemory=h5_inmemory, ANI1_like=True)


class ANINetworkDatasetForTetraPept(ANINetworkDataset):
    """Dataset class working with TetraPeptide dataset from Dr Duan for ANI Network
    """
    
    def __init__(
        self, df, h5_files, h5_multipole=None, h5_force=None, label_col=None, samp_w_col=None, shuffle=False, h5_cache=False, h5_inmemory=False, device="cpu"
    ) -> None:
        super().__init__(df, h5_files, h5_multipole=h5_multipole, h5_force=h5_force,
                         label_col=label_col, samp_w_col=samp_w_col, shuffle=shuffle, h5_cache=h5_cache, h5_inmemory=h5_inmemory, device=device)
        
    def set_h5db(self, h5_files, h5_cache, h5_inmemory):
        self.h5db = TetraPeptDatabaseH5(h5_file=h5_files[0], h5_cache=h5_cache, h5_inmemory=h5_inmemory, ANI1_like=True)


class RelativeDataset:
    """for relative energy training"""
    def __init__(self, dataset) -> None:
        # a variable storing the pairs
        self.dataset = dataset
        self.device = dataset.device
        self.pairs = self._sample_pairs()

    def _sample_pairs(self):
        """determinstic sampling method: (lowest_ene_conf, every_other_conf)"""
        df = self.dataset.df
        # TODO optimization: iterate over what groupby returns, maybe multiprocessing if needed.
        conf_counts = df.groupby("ID").apply(len)
        conf_counts = conf_counts[conf_counts > 1]
        
        pairs = []
        for mol_id in conf_counts.index:
            mol_df = df.query("ID == @mol_id")
            lowest_qm_conf_id = mol_df.loc[mol_df["RELATIVE_QM_ENERGY"].idxmin(), "CONF_ID"]
            pairs += [
                tuple(sorted(((mol_id, lowest_qm_conf_id), (mol_id, conf_id)), key=lambda x: x[1]))
                for conf_id in mol_df["CONF_ID"] if conf_id != lowest_qm_conf_id
            ]
        return pairs

    def __getitem__(self, idx):
        # support ((ID, CONF_ID), (ID, CONF_ID)) and numeric indices.
        # logger.debug(f"start to fetch table data for index {idx}")
        if type(idx) in (int, torch.int, torch.long):
            (mol1_id, conf1_id), (mol2_id, conf2_id) = self.pairs[idx]
        elif len(idx) == 2:
            (mol1_id, conf1_id), (mol2_id, conf2_id) = sorted(idx, key=lambda x: x[1])
        else:
            raise NotImplementedError(f"Unable to parse {idx} as valid index")
        data = {
            "the1st": self.dataset[mol1_id, conf1_id],
            "the2nd": self.dataset[mol2_id, conf2_id],
        }
        return data
        
    def __len__(self):
        return len(self.pairs)

    def __repr__(self) -> str:
        return str(self.pairs)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
            
    def collate_batch(self, batch):
        output = {k: self.dataset.collate_batch([data[k] for data in batch]) for k in ("the1st", "the2nd")}
        return output


class DatasetSpeedUp:
    """generate batches of training data
    """
    def __init__(self, dataset: ANINetworkDataset, batch_size=1, num_workers=1, shuffle=True, use_pkl="") -> None:
        self.device = dataset.device
        dataset.device = "cpu"

        if use_pkl and os.path.isfile(use_pkl):
            logger.info(f"Loading data from the pickled file {use_pkl}")
            self.data = pickle.load(open(use_pkl, "rb"))
            return
        # iterate over org dataset, move data to new h5
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset.collate_batch,
        )
        self.data = []
        for i, batch in enumerate(tqdm(loader)):
            self.data.append(batch)
            if i % 1000 == 0:
                logger.debug(f"loaded {i+1} batches; current memory usage:\n{os.popen('free -th; df -h').read()}")
            #     logger.debug(f"open files info: \n ulimit -n: {os.popen('ulimit -n').read().strip()} \n "
            #                 f"lsof | wc -l: {os.popen('lsof | wc -l').read().strip()} \n "
            #                 f"lsof -u yw24267 | wc -l: {os.popen('lsof -u yw24267 | wc -l').read().strip()} \n "
            #                 f"lsof -u yw24267 | grep \"dev/shm/torch\" | wc -l: {os.popen('lsof -u yw24267 | grep \"dev/shm/torch\" | wc -l').read().strip()} \n "
            #                 f"cat /proc/sys/fs/file-max: {os.popen('cat /proc/sys/fs/file-max').read().strip()}")
            # logger.debug(f"batch {i}, size {batch['energies'].shape}")
        if use_pkl:
            logger.info(f"Saving data to the pickled file {use_pkl}")
            pickle.dump(self.data, open(use_pkl, "wb"))
        
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(len(self)):  # num of batches
            yield self[i]


def get_dataset(dataset_name):
    """get dataset class

    Args:
        dataset_name (str): name for dataset classes

    Raises:
        NotImplementedError: cannot find the matching class

    Returns:
        mixed types: the matching dataset instance
    """
    dataset_class = {
        "ANI1": ANINetworkDataset,
        "ANI1x": ANINetworkDatasetForANI1x,
        "SPICE": ANINetworkDatasetForSPICE,
        "TetraPept": ANINetworkDatasetForTetraPept,
        "Combo": ANINetworkDatasetForTetraPept,
        "Metal": ANINetworkDatasetForMetal,
    }
    dataset_class = dataset_class.get(dataset_name, None)
    if dataset_class is None:
        raise NotImplementedError(f'{dataset_name} NOT supported!')
    return dataset_class
