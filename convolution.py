from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math


def conv2d(inputs, filters, strides, padding):
    """
    Performs 2D convolution given 4D inputs and filter Tensors.
    :param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
    :param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
    :param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
    :param padding: either "SAME" or "VALID", capitalization matters
    :return: outputs, NumPy array or Tensor with shape [num_examples, output_height, output_width, output_channels]
    """
    num_examples, in_height, in_width, input_in_channels = inputs.shape

    filter_height, filter_width, filter_in_channels, filter_out_channels = filters.shape
    num_examples_stride, strideY, strideX, channels_stride = strides

    assert input_in_channels == filter_in_channels
    padding_y = 0
    padding_x = 0
    # Cleaning padding input
    if padding == "SAME":
        padding_y = (filter_height - 1) / 2
        padding_x = (filter_width - 1) / 2

        y_paddings = [np.floor(padding_y), np.ceil(padding_y)]
        x_paddings = [np.floor(padding_x), np.ceil(padding_x)]

        paddings = np.array([[0, 0], y_paddings, x_paddings, [0, 0]]).astype(int)
        inputs = np.pad(inputs, pad_width=paddings, mode="constant", constant_values=0)
    # Calculate output dimensions
    output_height = int((in_height + 2 * padding_y - filter_height) / strideY + 1)
    output_width = int((in_width + 2 * padding_x - filter_width) / strideX + 1)
    output_array = np.zeros((num_examples, output_height, output_width, filter_out_channels))

    for i in range(num_examples):
        for in_c in range(input_in_channels):
            current_input = inputs[i, :, :, in_c]
            for y in range(output_height):
                for x in range(output_width):
                    for c in range(filter_out_channels):
                        output_array[i][y][x][c] += np.sum(current_input[y: y + filter_height, x: x + filter_width] * filters[:, :, in_c, c])
    return output_array


if __name__ == "__main__":
    imgs = np.array([[2, 2, 3, 3, 3], [0, 1, 3, 0, 3], [2, 3, 0, 1, 3], [3, 3, 2, 1, 2], [3, 3, 0, 2, 3], [2, 2, 3, 3, 3], [0, 1, 3, 0, 3], [2, 3, 0, 1, 3], [3, 3, 2, 1, 2], [3, 3, 0, 2, 3]],
                    dtype=np.float32)
    imgs = np.reshape(imgs, (1, 5, 5, 2))
    filters = tf.Variable(tf.random.truncated_normal([2, 2, 2, 2],
                                                     dtype=tf.float32,
                                                     stddev=1e-1),
                          name="filters")
    my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="SAME")
    print(my_conv)
    tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="SAME")
    print(tf_conv)
    print(my_conv == tf_conv)
