# written by Sean.Jie
# another
# ===============================
import numpy as np, tensorflow as tf
from tensorflow.python.util import nest



_state_size_with_prefix = tf.nn.rnn_cell._state_size_with_prefix

# a funtion to generate a zero tensor according different args
def zero_state(cell, batch_size, dtype):
    """Return zero-filled state tensor(s).
    Args:
      cell: RNNCell.
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.
    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.
      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
    the shapes `[batch_size x s]` for each s in `state_size`.
    """
    state_size = cell.state_size
    if nest.is_sequence(state_size):
        state_size_flat = nest.flatten(state_size)
        zeros_flat = [
            tf.zeros(
              tf.pack(_state_size_with_prefix(s, prefix=[batch_size])),
              dtype=dtype)
            for s in state_size_flat]
        for s, z in zip(state_size_flat, zeros_flat):
            z.set_shape(_state_size_with_prefix(s, prefix=[None]))
        zeros = nest.pack_sequence_as(structure=state_size,
                                    flat_sequence=zeros_flat)
    else:
        zeros_size = _state_size_with_prefix(state_size, prefix=[batch_size])
        zeros = tf.zeros(tf.pack(zeros_size), dtype=dtype)
        zeros.set_shape(_state_size_with_prefix(state_size, prefix=[None]))

    return zeros