# -*- coding: utf-8 -*-
# @Time       : 2022/05/19 17:42:29
# @Author     : Yanxing Wang <yxw@utexas.edu>
# @Project    : amoeba_nn
# @Description:

import time
import yaml
import logging
import collections.abc

__all__ = [
    "config",
    "load_yaml",
]

logger = logging.getLogger(__name__)


def recursive_update(d, u):
    """recursivly update a dictionary
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth

    Args:
        d (dict): dictionary being updated
        u (dict): dictionary storing updates

    Returns:
        dict: updated dictionary
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def load_yaml(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        new_conf = yaml.load(f, Loader=yaml.FullLoader)
    return new_conf


class Config:
    """Global configuration class for model training

    Returns:
        _type_: _description_
    """

    # default configs
    __conf = {
        "time_stamp": time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time())),
        "supported_species": ('H', 'C', 'N', 'O'),  # ordered by atomic numbers
        "aev": {
            'radial_start': 0.9, 'radial_cutoff': 5.2, 'radial_eta': 16.0, 'radial_dist_divisions': 8,
            'angular_start': 0.9, 'angular_cutoff': 3.5, 'angular_eta': 8.0, 'angular_dist_divisions': 4, 
            'zeta': 32.0, 'angle_sections': 8
        },
        "model": {
            "arch": 'ANINetwork',
            "MLP_dhiddens": (160, 128, 96),
        }
    }
    __conf["supported_species2idx"] = dict(zip(__conf["supported_species"], range(len(__conf["supported_species"]))))
    # __setters = ["device"]

    def to_dict(self):
        return self.__conf

    def __getitem__(self, name):
        return self.__conf[name]
    
    def get(self, name, default=None):
        return self.__conf.get(name, default)

    def __repr__(self):
        # return str(self.__conf)
        return yaml.dump(self.__conf, sort_keys=False)

    # def set(self, name, value):
    #     if name in self.__setters:
    #         self.__conf[name] = value
    #     else:
    #         raise NameError(f"Name \"{name}\" is immutable")

    def update(self, new_conf):
        self.__conf = recursive_update(self.__conf, new_conf)
        if "supported_species2idx" not in new_conf:
            self._update_species2idx_mapping()
        # logger.debug(f"updated config: {self.__conf}")

    def load(self, config_path):
        self.empty()
        new_conf = load_yaml(config_path)
        # logger.info(f'{"-"*20} config file: {config_path} loaded {"-"*20}')
        # logger.debug(f"initial config: {new_conf}")
        self.update(new_conf)

    def save(self, save_path):
        logger.info(f"runtime config: \n{self}")
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(self.__conf, f)
        logger.info(f'{"-"*20} config saved to {save_path} {"-"*20}')

    def empty(self):
        self.__conf = {
            "time_stamp": self.__conf["time_stamp"],
            "supported_species2idx": self.__conf["supported_species2idx"], 
            "supported_species": []
        }
        self.__conf["supported_species2idx"].clear()
    
    def to_tensorboard(self):
        return f"<pre>{self}</pre>"
    
    def _update_species2idx_mapping(self):
        self.__conf["supported_species2idx"].update(dict(zip(self["supported_species"], range(len(self["supported_species"])))))


config = Config()
