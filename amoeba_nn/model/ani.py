# -*- coding: utf-8 -*-
# @Time       : 2022/07/19 16:14:45
# @Author     : Yanxing Wang <yxw@utexas.edu>
# @Project    : amoeba_nn
# @Description: ANI models for benchmarking

from ase.optimize import LBFGS
from ase.constraints import FixInternals
from ase import Atoms
import torchani
import torch
from torch import nn
import logging
import numpy as np


logger = logging.getLogger(__name__)


class ANIWrapper(nn.Module):
    def __init__(self, name, **kwargs) -> None:
        super().__init__()
        model_dict = {
            "ANI1x": torchani.models.ANI1x,
            "ANI1ccx": torchani.models.ANI1ccx,
            "ANI2x": torchani.models.ANI2x,
        }
        self.model = model_dict[name](**kwargs)
    
    def forward(self, inputs) -> None:
        """forward function

        Args:
            inputs (Tuple[Tensor, Tensor]): (speices, coordinates)

        Returns:
            Tensor: total energies of molecules
        """
        output = self.model(inputs).energies
        return output

    def test_step(self, data):
        self.eval()
        with torch.no_grad():
            data = self.mv_data_to_device(data)
            xs, xc = data["species"], data["coordinates"]
            pred = self((xs, xc))
        return pred

    def predict(self, dataloader):
        preds = []
        for i, data in enumerate(dataloader):
            pred = self.test_step(data)
            preds.append(pred.cpu().numpy())
            if i % 100 == 0:
                logger.info(f"[ {i} / {len(dataloader)} ] batches completed.")
        preds = np.concatenate(preds)
        return preds

    def mv_data_to_device(self, data):
        # non_blocking
        for key in data.keys():
            data[key] = data[key].to(next(self.parameters()).device)
        return data
    
    def get_gradients(self, data):
        self.eval()
        data = self.mv_data_to_device(data)
        xs, xc = data["species"], data["coordinates"]
        xc.requires_grad = True
        ene = self((xs, xc))
        derivative = torch.autograd.grad(ene.sum(), xc)[0]
        return derivative

    def minimize(self, elems, xyz, torsions=(), threshold=0.01):
        molecule = Atoms(symbols=elems, positions=xyz)
        molecule.set_calculator(self.model.cpu().ase())

        ls=[]
        for indices in torsions: 
            dihedral = [molecule.get_dihedral(*indices), indices] 
            ls.append(dihedral)
        c = FixInternals(dihedrals_deg=ls)
        molecule.set_constraint(c)

        opt = LBFGS(molecule, logfile="/dev/null")
        opt.run(fmax=threshold)
        pos = opt.atoms.get_positions().tolist()
        ene = opt.atoms.get_potential_energy()
        ev2ha = 0.0367493  # eV to Hartree
        ene *= ev2ha
        return opt.converged(), ene, pos
