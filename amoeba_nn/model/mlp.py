# -*- coding: utf-8 -*-
# @Time       : 2022/05/18 17:01:06
# @Author     : Yanxing Wang <yxw@utexas.edu>
# @Project    : amoeba_nn
# @Description: Model classes for AMOEBA+NN models

import numpy as np
import torch
from torch import nn
import logging
from itertools import combinations_with_replacement

from .aev import cover_linearly
from ..data.dataset import ANINetworkDataset
from ..utils.config import config
from ..utils.helpers import Regressor_Performance


logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """MLP building block class"""
    def __init__(self, din=100, dout=1, dhiddens=(), act_layer=nn.CELU, bias=True, last_layer=None, batch_norm=False) -> None:
        super().__init__()
        in_dims = [din] + list(dhiddens)[:-1]
        out_dims = list(dhiddens)
        layers = []
        for i, o in zip(in_dims, out_dims):
            layers.append(nn.Linear(i, o, bias=bias))
            if batch_norm:
                layers.append(nn.BatchNorm1d(o))
            layers.append(act_layer())
        layers.append(nn.Linear(out_dims[-1], dout, bias=bias))
        if last_layer:
            layers.append(last_layer)
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)


class ANINetwork(nn.Module):
    """Base model class"""
    def __init__(self, force_weight=0) -> None:
        self.enable_bias = config['model'].get("enable_bias", True)
        super().__init__()

        self.set_aev_computer()
        self.set_potential_networks()

        self.loss = None
        self.optimizer = None
        self.scheduler = None
        self.is_force_enabled = bool(force_weight)
        self.force_weight = force_weight
        self.is_training = False

    def set_aev_computer(self):
        # AEV calculator
        self.aev_computer = cover_linearly(**config['aev'], num_species=len(config["supported_species"]))
        # workaround to create the computer with cuda ext, simply because of .cover_linerly giving an instance without cuda ext.
        # but seems like my installation of aev cuda ext has problems...
        # Rcr, EtaR, ShfR, Rca, ShfZ, EtaA, Zeta, ShfA = aev_computer.constants()
        # aev_computer = AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species=4, use_cuda_extension=True).cuda()

    def set_potential_networks(self):
        # network for each element
        self.potential_networks = nn.ModuleList([
            MLP(
                din=self.aev_computer.aev_length,
                dhiddens=config['model']["MLP_dhiddens"],
                bias=self.enable_bias,
            )
            for _ in range(len(config['supported_species']))
        ])

    def set_species(self, species):
        # for testgrad, should be removed later
        self.xs = species

    def forward(self, inputs) -> None:
        """forward function

        Args:
            inputs (Tuple[Tensor, Tensor]): (speices, coordinates)

        Returns:
            Tensor: total energies of molecules
        """
        # adopted from torchANI
        # this if statement is for testgrad, should be removed later
        if isinstance(inputs, torch.Tensor):
            inputs = (self.xs, inputs)
        elif isinstance(inputs, tuple):
            pass
        else:
            raise TypeError("Input type not supported.")
        
        if self.is_force_enabled:  # require second order gradient for force evaluation / training
            inputs = (inputs[0], inputs[1].requires_grad_(True))
        species, aev = self.forward_aev_part(inputs)
        enes = self.forward_potential_part(species, aev)
        # print("requires_grad:", enes.requires_grad, aev.requires_grad, species.requires_grad, flush=True)
        if self.is_force_enabled:  # eval force
            fs = -torch.autograd.grad(enes.sum(), inputs[1], create_graph=self.is_training, retain_graph=self.is_training)[0]
            return enes, fs
        else:
            return enes
    
    def forward_aev_part(self, inputs):
        species, aev = self.aev_computer(inputs)
        assert species.shape == aev.shape[:-1]
        return species, aev

    def forward_potential_part(self, species, aev):
        species_ = species.flatten()
        aev = aev.flatten(0, 1)
        output = aev.new_zeros(species_.shape)
        for i, m in enumerate(self.potential_networks):
            mask = (species_ == i)
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output.masked_scatter_(mask, m(input_).flatten())
        output = output.view_as(species).sum(dim=-1)
        return output

    def set_loss(self, loss):
        self.loss = loss

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def scheduler_step(self, loss, epoch_idx, writer):
        if self.scheduler:
            self.scheduler.step(loss)
            writer.add_scalar("Parameters/learning_rate", self.optimizer.param_groups[0]['lr'], epoch_idx)

    def train_step(self, data):
        self.is_training = True
        self.train()
        for param in self.parameters():
            param.requires_grad = self.is_training
        self.optimizer.zero_grad()

        data = self.mv_data_to_device(data)
        xs, xc, y, w = data["species"], data["coordinates"], data["energies"], data["sample_weights"]
        logger.debug(f"y: {y.shape}")
        logger.debug(f"y dtype: {y.dtype}")
        pred = self((xs, xc))
        if self.is_force_enabled:
            pred, pred_f = pred
        logger.debug(f"pred dtype: {pred.dtype}")
        loss_e = self.loss(pred, y, weight=w)
        loss = loss_e
        logger.debug(f"loss dtype: {type(loss)}, {loss.dtype}")
        if self.is_force_enabled:
            f = data["forces"]
            logger.debug(f"f: {f}")
            logger.debug(f"f: {f.shape}")
            logger.debug(f"pred_f: {pred_f.shape}")
            loss_f = self.loss(pred_f, f, weight=w, for_forces=config["train"]["loss_fn_force"], reduction=True)
            loss = loss + loss_f * self.force_weight  # weird that loss "+=" loss_f * self.force_weight does not work
            logger.debug(f"loss: {loss.item()}, loss_e: {loss_e.item()}, loss_f: {loss_f.item()}, force_weight: {self.force_weight}")

        loss.backward()
        self.optimizer.step()
        logger.debug(f"parameters updated after a step. e.g., {next(self.parameters())}")
        if self.is_force_enabled:
            return loss.item(), loss_e.item(), loss_f.item()
        else:
            return loss.item()
        
    def train_epoch(self, dataloader=None, epoch_idx=None, writer=None):
        assert all([self.optimizer, self.loss]), "optimizer / loss not set!"
        # loss_epoch = []
        for batch_idx, data in enumerate(dataloader):
            logger.debug("batch of training data loaded.")
            loss = self.train_step(data)
            logger.debug(f"parameters updated after an epoch. e.g., {next(self.parameters())}")
            if self.is_force_enabled:
                loss, loss_e, loss_f = loss
                writer.add_scalar("Loss/step_force", loss_f, epoch_idx * len(dataloader) + batch_idx)
                writer.add_scalar("Loss/step_energy", loss_e, epoch_idx * len(dataloader) + batch_idx)
            logger.debug("train_step completed.")
            logger.debug(f"loss: {loss}, step: {epoch_idx * len(dataloader) + batch_idx}")
            writer.add_scalar("Loss/step", loss, epoch_idx * len(dataloader) + batch_idx)
            logger.debug("training loss logged to tensorboard.")

            # loss_epoch.append(loss)
        # loss_epoch = torch.cat(loss_epoch).mean()  # possible subtle error due to the smaller last batch
        # writer.add_scalar("Loss/epoch", loss, epoch_idx * len(dataloader) + batch_idx)
        # return loss_epoch

    def test_step(self, data):
        self.is_training = False
        self.eval()
        for param in self.parameters():
            param.requires_grad = self.is_training
        # with torch.no_grad():  not compatible with testing forces.

        data = self.mv_data_to_device(data)
        xs, xc, y, w = data["species"], data["coordinates"], data["energies"], data["sample_weights"]
        logger.debug(f"y: {y.shape}")
        pred = self((xs, xc))

        if self.is_force_enabled:
            pred, pred_f = pred
            f = data["forces"]
            return y, pred, pred_f, f, w
        else:
            return y, pred, w 
        
    def test_epoch(self, dataloader=None, epoch_idx=None, writer=None):
        ys, preds, ws = [], [], []
        if self.is_force_enabled:
            loss_f = 0
            loss_f_num_samples = 0

        for _, data in enumerate(dataloader):
            logger.debug("batch of test data loaded.")
            out = self.test_step(data)
            if self.is_force_enabled:
                y, pred, pred_f, f, w = out
                loss_f_i, nums_f_i = self.loss(pred_f, f, weight=w, for_forces=config["train"]["loss_fn_force"], reduction=False)
                loss_f += loss_f_i
                loss_f_num_samples += nums_f_i
            else:
                y, pred, w = out
            ws.append(w.cpu().numpy())
            logger.debug("test_step completed.")
            ys.append(y.cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
            logger.debug("test labels and predictions appended.")

        preds, ys, ws = np.concatenate(preds), np.concatenate(ys), np.concatenate(ws)
        loss_e = self.loss(torch.from_numpy(preds), torch.from_numpy(ys), weight=torch.from_numpy(ws))
        if self.is_force_enabled:
            loss_f = loss_f / loss_f_num_samples
            loss = loss_e + loss_f * self.force_weight
            writer.add_scalar("Loss/force_weight", self.force_weight, epoch_idx)
            writer.add_scalar("Loss/epoch_force", loss_f, epoch_idx)
            writer.add_scalar("Loss/epoch_energy", loss_e, epoch_idx)
        else:
            loss = loss_e
        writer.add_scalar("Loss/epoch", loss, epoch_idx)
        perfs = Regressor_Performance(ys, preds).get_all_as_dict()
        writer.add_scalar("Performance/RMSE_epoch", perfs["RMSE"], epoch_idx)
        writer.add_scalar("Performance/R_Pearson_epoch", perfs["R_Pearson"], epoch_idx)
        writer.add_scalar("Performance/R_Spearman_epoch", perfs["R_Spearman"], epoch_idx)
        writer.add_scalar("Performance/R_squared_epoch", perfs["R_Squared"], epoch_idx)
        perfs = {"Epoch_ID": epoch_idx, "Loss": loss.item(), **perfs}
        return perfs

    def predict(self, dataloader):
        preds = []
        for i, data in enumerate(dataloader):
            pred = self.predict_step(data)
            preds.append(pred.cpu().numpy())
            if i % 100 == 0:
                logger.info(f"[ {i} / {len(dataloader)} ] batches completed.")
        preds = np.concatenate(preds)
        return preds
    
    def predict_step(self, data):
        """just energy prediction"""
        self.eval()
        with torch.no_grad():
            data = self.mv_data_to_device(data)
            xs, xc = data["species"], data["coordinates"]
            # disable force eval temporarily for energy only prediction
            is_force_enabled = self.is_force_enabled
            self.is_force_enabled = False
            pred = self((xs, xc))
            self.is_force_enabled = is_force_enabled
        return pred

    def analyze(self, data):
        """calculate both energy and gradient"""
        self.eval()
        self.is_force_enabled = True
        for p in self.parameters():
            p.requires_grad = False
        data = self.mv_data_to_device(data)
        xs, xc = data["species"], data["coordinates"]
        enes, fs = ANINetwork.forward(self, (xs, xc))
        grads = -fs
        self.is_force_enabled = False
        return enes.cpu(), grads.cpu()

    def mv_data_to_device(self, data):
        # non_blocking
        for key in data.keys():
            data[key] = data[key].to(next(self.parameters()).device)
        return data
    

class ANINetwork_Metal(ANINetwork):
    """model for metal ions"""
    def __init__(self, *args, **kwargs):
        # set up metal's info
        self.metal_symbol = config["model"]["metal"]
        self.metal_species_num = ANINetworkDataset.supported_species[self.metal_symbol]
        super().__init__(*args, **kwargs)

    def set_aev_metal_mask(self):
        # find the mask for masking out the metal related components in AEV. 
        # this is a tentative workaround to make use of the AEVComputer in torchANI for metal.
        num_species, radial_sublength, radial_length, angular_sublength, angular_length = self.aev_computer.sizes
        rad_mask = [int(self.metal_species_num * radial_sublength + i) for i in range(radial_sublength)]
        ang_mask = []
        for p in combinations_with_replacement(range(num_species), 2):
            if self.metal_species_num in p:
                spj, spk = p  # note: spj has to be less than or equal to spk
                ang_mask += [
                    int(radial_length + (spj * (2 * num_species - spj - 1) / 2 + spk) * angular_sublength + i)
                    for i in range(angular_sublength)
                ]
        self.aev_metal_mask = torch.Tensor([i for i in range(self.aev_computer.aev_length) if i not in rad_mask + ang_mask]).to(torch.long)
        
    def set_potential_networks(self):
        # network for the metal, only one is enough
        # need to manually compute the input dimension, because the compatability of AEV computer
        self.set_aev_metal_mask()
        self.potential_networks = nn.ModuleList([
            MLP(
                din=len(self.aev_metal_mask),
                dhiddens=config['model']["MLP_dhiddens"],
                bias=self.enable_bias,
            )
        ])
    
    def forward_aev_part(self, inputs):
        species, aev = self.aev_computer(inputs)
        assert species.shape == aev.shape[:-1]
        aev = aev[:, :, self.aev_metal_mask]
        return species, aev

    def forward_potential_part(self, species, aev):
        species_ = species.flatten()
        aev = aev.flatten(0, 1)
        output = aev.new_zeros(species_.shape)
        mask = (species_ == self.metal_species_num)
        midx = mask.nonzero().flatten()
        if midx.shape[0] > 0:
            input_ = aev.index_select(0, midx)
            output.masked_scatter_(mask, self.potential_networks[0](input_).flatten())
        # print(output.view_as(species))
        output = output.view_as(species).sum(dim=-1)
        return output


class ANINetwork_Relative(ANINetwork):
    """class for relative energy training"""
    def __init__(self, fit_both_rel_abs=False, **kwargs) -> None:
        self.fit_both_rel_abs = fit_both_rel_abs
        super().__init__(**kwargs)

    def forward(self, inputs):
        """forward function for a pair of conf
        """
        inp1, inp2 = inputs
        ene1, ene2 = super().forward(inp1), super().forward(inp2)
        if self.is_force_enabled:  
            ene1, f1 = ene1
            ene2, f2 = ene2
        diff = ene1 - ene2
        # logger.debug(f"inp1[0].requires_grad: {inp1[0].requires_grad}, inp1[1].requires_grad: {inp1[1].requires_grad}")
        # f1 = -torch.autograd.grad(ene1.sum(), inp1[1], create_graph=self.is_training, retain_graph=self.is_training)[0]
        # f2 = -torch.autograd.grad(ene2.sum(), inp2[1], create_graph=self.is_training, retain_graph=self.is_training)[0]
        if self.fit_both_rel_abs:
            if self.is_force_enabled:
                return (diff, ene1, ene2), f1, f2
            else:
                return diff, ene1, ene2
        else:
            if self.is_force_enabled:
                return diff, f1, f2
            else:
                return diff

    def train_step(self, pair):
        self.is_training = True
        self.train()
        for param in self.parameters():
            param.requires_grad = self.is_training
        self.optimizer.zero_grad()

        pair = self.mv_pair_to_device(pair)
        data1, data2 = pair["the1st"], pair["the2nd"]
        xs1, xc1, y1, w1 = data1["species"], data1["coordinates"], data1["energies"], data1["sample_weights"]
        xs2, xc2, y2, w2 = data2["species"], data2["coordinates"], data2["energies"], data2["sample_weights"]
        y = y1 - y2
        ### seems unnecessary after the refactoring
        # if self.is_force_enabled:  # require second order gradient
        #     xc1, xc2 = xc1.requires_grad_(True), xc2.requires_grad_(True)
        logger.debug(f"y: {y.shape}")
        logger.debug(f"y dtype: {y.dtype}")
        pred = self(((xs1, xc1), (xs2, xc2)))
        logger.debug(f"is_force_enabled: {self.is_force_enabled}, fit_both_rel_abs: {self.fit_both_rel_abs}")            
        if self.fit_both_rel_abs:
            if self.is_force_enabled:
                pred, pred_f1, pred_f2 = pred
                pred, pred_e1, pred_e2 = pred
            else:
                pred, pred_e1, pred_e2 = pred
        else:
            if self.is_force_enabled:
                pred, pred_f1, pred_f2 = pred
            else:
                pred = pred
        logger.debug(f"pred dtype: {pred.dtype}")
        loss_e = self.loss(pred, y, weight=w1)
        if self.fit_both_rel_abs:
            loss_e1 = self.loss(pred_e1, y1, weight=w1)
            loss_e2 = self.loss(pred_e2, y2, weight=w2)
            loss_e = (loss_e + loss_e1 + loss_e2) / 3  # average the losses
        loss = loss_e
        logger.debug(f"loss dtype: {type(loss)}, {loss.dtype}")
        if self.is_force_enabled:
            f1, f2 = data1["forces"], data2["forces"]
            logger.debug(f"f1: {f1}, f2: {f2}")
            logger.debug(f"f1: {f1.shape}, f2: {f2.shape}")
            logger.debug(f"pred_f1: {pred_f1.shape}, pred_f2: {pred_f2.shape}")
            loss_f1, nums_f1 = self.loss(pred_f1, f1, weight=w1, for_forces=config["train"]["loss_fn_force"], reduction=False)
            loss_f2, nums_f2 = self.loss(pred_f2, f2, weight=w2, for_forces=config["train"]["loss_fn_force"], reduction=False)
            loss_f = (loss_f1 + loss_f2) / (nums_f1 + nums_f2)
            loss = loss + loss_f * self.force_weight  # weird that loss "+=" loss_f * self.force_weight does not work
            logger.debug(f"loss: {loss.item()}, loss_e: {loss_e.item()}, loss_f: {loss_f.item()}, force_weight: {self.force_weight}")

        loss.backward()
        self.optimizer.step()
        logger.debug(f"parameters updated after a step. e.g., {next(self.parameters())}")
        if self.is_force_enabled:
            return loss.item(), loss_e.item(), loss_f.item()
        else:
            return loss.item()

    def test_step(self, pair):
        self.is_training = False
        self.eval()
        for param in self.parameters():
            param.requires_grad = self.is_training
        # with torch.no_grad():  not compatible with testing forces.

        pair = self.mv_pair_to_device(pair)
        data1, data2 = pair["the1st"], pair["the2nd"]
        xs1, xc1, y1, w1 = data1["species"], data1["coordinates"], data1["energies"], data1["sample_weights"]
        xs2, xc2, y2, w2 = data2["species"], data2["coordinates"], data2["energies"], data2["sample_weights"]
        y = y1 - y2
        logger.debug(f"y: {y.shape}")
        ### seems unnecessary after the refactoring
        # if self.is_force_enabled:  # require second order gradient
        #     xc1, xc2 = xc1.requires_grad_(True), xc2.requires_grad_(True)
        pred = self(((xs1, xc1), (xs2, xc2)))

        if self.is_force_enabled:
            pred, pred_f1, pred_f2 = pred
            f1, f2 = data1["forces"], data2["forces"]
            if self.fit_both_rel_abs:
                pred, pred_e1, pred_e2 = pred
                return (y, y1, y2), (pred, pred_e1, pred_e2), pred_f1, pred_f2, f1, f2, w1, w2
            return y, pred, pred_f1, pred_f2, f1, f2, w1, w2
        else:
            if self.fit_both_rel_abs:
                pred, pred_e1, pred_e2 = pred
                return (y, y1, y2), (pred, pred_e1, pred_e2), w1, w2
            return y, pred, w1, w2
        
    def test_epoch(self, dataloader=None, epoch_idx=None, writer=None):
        ys, preds = [], []
        w1s, w2s = [], []
        if self.fit_both_rel_abs:
            ys1, ys2, preds_e1, preds_e2 = [], [], [], []
        if self.is_force_enabled:
            loss_f = 0
            loss_f_num_samples = 0
        for _, data in enumerate(dataloader):
            logger.debug("batch of test data loaded.")
            out = self.test_step(data)
            if self.is_force_enabled:
                y, pred, pred_f1, pred_f2, f1, f2, w1, w2 = out
                if self.fit_both_rel_abs:
                    y, y1, y2 = y
                    pred, pred_e1, pred_e2 = pred
                loss_f1, nums_f1 = self.loss(pred_f1, f1, weight=w1, for_forces=config["train"]["loss_fn_force"], reduction=False)
                loss_f2, nums_f2 = self.loss(pred_f2, f2, weight=w2, for_forces=config["train"]["loss_fn_force"], reduction=False)
                loss_f += loss_f1 + loss_f2
                loss_f_num_samples += nums_f1 + nums_f2
            else:
                y, pred, w1, w2 = out
                if self.fit_both_rel_abs:
                    y, y1, y2 = y
                    pred, pred_e1, pred_e2 = pred
            logger.debug("test_step completed.")
            w1s.append(w1.cpu().numpy())
            w2s.append(w2.cpu().numpy())
            ys.append(y.cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
            if self.fit_both_rel_abs:
                ys1.append(y1.cpu().numpy())
                ys2.append(y2.cpu().numpy())
                preds_e1.append(pred_e1.detach().cpu().numpy())
                preds_e2.append(pred_e2.detach().cpu().numpy())
            logger.debug("test labels and predictions appended.")

        preds, ys = np.concatenate(preds), np.concatenate(ys)
        w1s, w2s = np.concatenate(w1s), np.concatenate(w2s)
        if self.fit_both_rel_abs:
            ys1, ys2 = np.concatenate(ys1), np.concatenate(ys2)
            preds_e1, preds_e2 = np.concatenate(preds_e1), np.concatenate(preds_e2)
        if self.is_force_enabled:
            loss_e = self.loss(torch.from_numpy(preds), torch.from_numpy(ys), weight=torch.from_numpy(w1s))
            if self.fit_both_rel_abs:
                loss_e1 = self.loss(torch.from_numpy(preds_e1), torch.from_numpy(ys1), weight=torch.from_numpy(w1s))
                loss_e2 = self.loss(torch.from_numpy(preds_e2), torch.from_numpy(ys2), weight=torch.from_numpy(w2s))
                loss_e = (loss_e + loss_e1 + loss_e2) / 3
            loss_f = loss_f / loss_f_num_samples
            loss = loss_e + loss_f * self.force_weight
            writer.add_scalar("Loss/force_weight", self.force_weight, epoch_idx)
            writer.add_scalar("Loss/epoch_force", loss_f, epoch_idx)
            writer.add_scalar("Loss/epoch_energy", loss_e, epoch_idx)
        else:
            loss = self.loss(torch.from_numpy(preds), torch.from_numpy(ys), weight=torch.from_numpy(w1s))
            if self.fit_both_rel_abs:
                loss_e1 = self.loss(torch.from_numpy(preds_e1), torch.from_numpy(ys1), weight=torch.from_numpy(w1s))
                loss_e2 = self.loss(torch.from_numpy(preds_e2), torch.from_numpy(ys2), weight=torch.from_numpy(w2s))
                loss = (loss + loss_e1 + loss_e2) / 3
        writer.add_scalar("Loss/epoch", loss, epoch_idx)
        perfs = Regressor_Performance(ys, preds).get_all_as_dict()
        writer.add_scalar("Performance/RMSE_epoch", perfs["RMSE"], epoch_idx)
        writer.add_scalar("Performance/R_Pearson_epoch", perfs["R_Pearson"], epoch_idx)
        writer.add_scalar("Performance/R_Spearman_epoch", perfs["R_Spearman"], epoch_idx)
        writer.add_scalar("Performance/R_squared_epoch", perfs["R_Squared"], epoch_idx)
        perfs = {"Epoch_ID": epoch_idx, "Loss": loss.item(), **perfs}
        if self.fit_both_rel_abs:
            logger.debug("writing performance of individuals in the pair.")
            perfs1 = Regressor_Performance(ys1, preds_e1).get_all_as_dict()
            perfs2 = Regressor_Performance(ys2, preds_e2).get_all_as_dict()
            writer.add_scalar("Performance/Ene1_RMSE_epoch", perfs1["RMSE"], epoch_idx)
            writer.add_scalar("Performance/Ene2_RMSE_epoch", perfs2["RMSE"], epoch_idx)
            writer.add_scalar("Performance/Ene1_R_Pearson_epoch", perfs1["R_Pearson"], epoch_idx)
            writer.add_scalar("Performance/Ene2_R_Pearson_epoch", perfs2["R_Pearson"], epoch_idx)
            writer.add_scalar("Performance/Ene1_R_Spearman_epoch", perfs1["R_Spearman"], epoch_idx)
            writer.add_scalar("Performance/Ene2_R_Spearman_epoch", perfs2["R_Spearman"], epoch_idx)
            writer.add_scalar("Performance/Ene1_R_squared_epoch", perfs1["R_Squared"], epoch_idx)
            writer.add_scalar("Performance/Ene2_R_squared_epoch", perfs2["R_Squared"], epoch_idx)
            perfs.update({"Ene1_" + k: v for k, v in perfs1.items()})
            perfs.update({"Ene2_" + k: v for k, v in perfs2.items()})
        return perfs

    def mv_pair_to_device(self, pair):
        # non_blocking
        for pairkey, data in pair.items():
            for key in data.keys():
                pair[pairkey][key] = pair[pairkey][key].to(next(self.parameters()).device)
        return pair
     

class ANINetwork_pos(ANINetwork):
    """model that only outputs positive energies"""
    def __init__(self) -> None:
        super().__init__()        
        # network for each element
        self.potential_networks = nn.ModuleList([
            MLP(
                din=self.aev_computer.aev_length,
                dhiddens=config['model']["MLP_dhiddens"],
                last_layer=nn.Softplus(),
            )
            for _ in range(len(config['supported_species']))
        ])


class ANINetwork_BN(ANINetwork):
    """model with batchnorm layers"""
    def __init__(self) -> None:
        super().__init__()        
        # network for each element
        self.potential_networks = nn.ModuleList([
            MLP(
                din=self.aev_computer.aev_length,
                dhiddens=config['model']["MLP_dhiddens"],
                batch_norm=True,
            )
            for _ in range(len(config['supported_species']))
        ])


class ANINetwork_Multipole(ANINetwork):
    """model that takes AEV + atomic multipole moments as input"""
    def __init__(self) -> None:
        super().__init__()
        # network for each element
        self.potential_networks = nn.ModuleList([
            MLP(
                din=self.aev_computer.aev_length + 10,
                dhiddens=config['model']["MLP_dhiddens"],
            )
            for _ in range(len(config['supported_species']))
        ])

    def forward(self, inputs) -> None:
        """forward function

        Args:
            inputs (Tuple[Tensor, Tensor]): (speices, coordinates)

        Returns:
            Tensor: total energies of molecules
        """
        species, coords, multipoles = inputs
        # adopted from torchANI
        species, aev = self.aev_computer((species, coords))
        assert species.shape == aev.shape[:-1]
        species_ = species.flatten()
        in_feature = torch.cat([aev, multipoles], dim=-1)
        in_feature = in_feature.flatten(0, 1)

        output = in_feature.new_zeros(species_.shape)

        for i, m in enumerate(self.potential_networks):
            mask = (species_ == i)
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = in_feature.index_select(0, midx)
                output.masked_scatter_(mask, m(input_).flatten())
        output = output.view_as(species).sum(dim=-1)
        return output

    def train_step(self, data):
        self.train()
        self.optimizer.zero_grad()
        data = self.mv_data_to_device(data)
        xs, xc, xm, y = data["species"], data["coordinates"], data["multipoles"], data["energies"]
        logger.debug(f"y: {y.shape}")
        logger.debug(f"y dtype: {y.dtype}")
        pred = self((xs, xc, xm))
        logger.debug(f"pred dtype: {pred.dtype}")
        loss = self.loss(pred, y)
        logger.debug(f"loss dtype: {type(loss)}, {loss.dtype}")
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test_step(self, data):
        self.eval()
        with torch.no_grad():
            data = self.mv_data_to_device(data)
            xs, xc, xm, y, w = data["species"], data["coordinates"], data["multipoles"], data["energies"], data["sample_weights"]
            logger.debug(f"y: {y.shape}")
            pred = self((xs, xc, xm))
        return y, pred, w


class ANINetwork_MonoDipole(ANINetwork_Multipole):
    """model that takes AEV + Monopole + Dipole moments as input"""
    def __init__(self) -> None:
        super().__init__()
        # network for each element
        self.potential_networks = nn.ModuleList([
            MLP(
                din=self.aev_computer.aev_length + 2,
                dhiddens=config['model']["MLP_dhiddens"],
            )
            for _ in range(len(config['supported_species']))
        ])

    def forward(self, inputs) -> None:
        """forward function

        Args:
            inputs (Tuple[Tensor, Tensor]): (speices, coordinates)

        Returns:
            Tensor: total energies of molecules
        """
        species, coords, multipoles = inputs
        mono_dipole = torch.stack([multipoles[:, :, 0], (multipoles[:, :, 1:4] ** 2).sum(-1)], dim=-1)
        # adopted from torchANI
        species, aev = self.aev_computer((species, coords))
        assert species.shape == aev.shape[:-1]
        species_ = species.flatten()
        in_feature = torch.cat([aev, mono_dipole], dim=-1)
        in_feature = in_feature.flatten(0, 1)

        output = in_feature.new_zeros(species_.shape)

        for i, m in enumerate(self.potential_networks):
            mask = (species_ == i)
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = in_feature.index_select(0, midx)
                output.masked_scatter_(mask, m(input_).flatten())
        output = output.view_as(species).sum(dim=-1)
        return output


class ANINetwork_Metal_Relative(ANINetwork_Metal, ANINetwork_Relative):
    # TODO separate out relative training logic as a standalone class not inheriting from ANINetwork, so the multiple inheritance is easier and more clear.
    """class for relative energy training with metal-specific AEV processing"""
    def __init__(self, fit_both_rel_abs=True, **kwargs) -> None:
        ANINetwork_Metal.__init__(self, **kwargs)
        self.fit_both_rel_abs = fit_both_rel_abs  # has to be set after ANINetwork_Metal.__init__() to avoid it being overwritten by ANINetwork_Relative.__init__()
        logger.debug(f"using ANINetwork_Metal_Relative, fit_both_rel_abs: {self.fit_both_rel_abs}")
