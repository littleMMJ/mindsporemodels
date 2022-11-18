"""
python initializer.py
"""
import math
import numpy as np

from mindspore.common.initializer import Initializer, _calculate_fan_in_and_fan_out, _assignment, _register

_INITIALIZER_ALIAS = dict()

@_register('glorot_normal')
class GlorotNormal(Initializer):
    r"""
    Initialize the array with Glorot Normal algorithm, and from a normal distribution collect samples within
    N(0, std), The std is defined as:

    .. math::
        std = gain * \sqrt{\frac{2}{n_{in} + n_{out}}}

    - where :math:`n_{in}` is the number of input units in the weight tensor.
    - where :math:`n_{out}` is the number of output units in the weight tensor.

    Args:
        gain (float): An optional scaling factor. Default: 1.

    Returns:
        Array, assigned array.
    """

    def __init__(self, gain=1):
        super(GlorotNormal, self).__init__(gain=gain)
        self.gain = gain

    def _initialize(self, arr):
        n_in, n_out = _calculate_fan_in_and_fan_out(arr.shape)

        std = self.gain * math.sqrt(2.0 / (n_in + n_out))
        data = np.random.normal(0, std, arr.shape)

        _assignment(arr, data)
