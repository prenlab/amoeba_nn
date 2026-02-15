# -*- coding: utf-8 -*-
# @Time       : 2023/01/25 18:11:01
# @Author     : Yanxing Wang <yxw@utexas.edu>
# @Project    : amoeba_nn
# @Description: AEV Calculator

import torch
from torchani import AEVComputer


# not sure the reason but the torchani from conda-forge doesn't come with this class method.
def cover_linearly(radial_cutoff: float, angular_cutoff: float,
                    radial_eta: float, angular_eta: float,
                    radial_dist_divisions: int, angular_dist_divisions: int,
                    zeta: float, angle_sections: int, num_species: int,
                    angular_start: float = 0.9, radial_start: float = 0.9):
    r""" Provides a convenient way to linearly fill cutoffs
    This is a user friendly constructor that builds an
    :class:`torchani.AEVComputer` where the subdivisions along the the
    distance dimension for the angular and radial sub-AEVs, and the angle
    sections for the angular sub-AEV, are linearly covered with shifts. By
    default the distance shifts start at 0.9 Angstroms.
    To reproduce the ANI-1x AEV's the signature ``(5.2, 3.5, 16.0, 8.0, 16, 4, 32.0, 8, 4)``
    can be used.
    """
    # This is intended to be self documenting code that explains the way
    # the AEV parameters for ANI1x were chosen. This is not necessarily the
    # best or most optimal way but it is a relatively sensible default.
    Rcr = radial_cutoff
    Rca = angular_cutoff
    EtaR = torch.tensor([float(radial_eta)])
    EtaA = torch.tensor([float(angular_eta)])
    Zeta = torch.tensor([float(zeta)])

    ShfR = torch.linspace(radial_start, radial_cutoff, radial_dist_divisions + 1)[:-1]
    ShfA = torch.linspace(angular_start, angular_cutoff, angular_dist_divisions + 1)[:-1]
    angle_start = torch.pi / (2 * angle_sections)

    ShfZ = (torch.linspace(0, torch.pi, angle_sections + 1) + angle_start)[:-1]

    return AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
