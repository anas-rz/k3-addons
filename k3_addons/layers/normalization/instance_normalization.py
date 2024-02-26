import logging

from keras.layers import GroupNormalization
from k3_addons.api_export import k3_export


@k3_export(path="k3_addons.layers.InstanceNormalization")
class InstanceNormalization(GroupNormalization):
    def __init__(self, **kwargs):
        if "groups" in kwargs:
            logging.warning("The given value for groups will be overwritten.")
        super().__init__(groups=1, **kwargs)
