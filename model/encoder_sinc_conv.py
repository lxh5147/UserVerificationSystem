import math

import numpy as np
import tensorflow as tf


def sinc(band, t_right):
    y_right = tf.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)
    y_left = tf.reverse(y_right, [0])
    y = tf.concat([y_left, [1.], y_right], axis=0)
    return y


def encoder(inputs,
            params,
            upper_encoder,
            is_training=True
            ):
    # Mel Initialization of the filter banks
    freq_scale = params['freq_scale']
    filter_number = params['sinc_filters']  # the number of filters
    kernel_size = params['sinc_kernel_size']  # the filter kernel size
    low_freq_mel = 80
    high_freq_mel = (2595 * np.log10(1 + (freq_scale / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, filter_number)  # Equally spaced in Mel scale
    f_cos = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    b1 = np.roll(f_cos, 1)
    b2 = np.roll(f_cos, -1)
    b1[0] = 30
    b2[-1] = (freq_scale / 2) - 100
    with tf.variable_scope("encoder"):
        filter_b1 = tf.get_variable("filter_b1", initializer=tf.constant(b1 / freq_scale, dtype='float32'))
        filter_band = tf.get_variable("filter_band", initializer=tf.constant((b2 - b1) / freq_scale, dtype='float32'))
        t_right = tf.linspace(1., (kernel_size - 1) / 2, num=int((kernel_size - 1) / 2)) / freq_scale
        min_freq = 50.0
        min_band = 50.0
        filter_freq_beg = tf.abs(filter_b1) + min_freq / freq_scale
        filter_freq_end = filter_freq_beg + (tf.abs(filter_band) + min_band / freq_scale)
        n = tf.linspace(0., float(kernel_size), num=kernel_size)
        # Filter window (hamming)
        window = 0.54 - 0.46 * tf.cos(2 * math.pi * n / kernel_size)
        filters = []
        for i in range(filter_number):
            low_pass1 = 2 * filter_freq_beg[i] * sinc(filter_freq_beg[i] * freq_scale, t_right)
            low_pass2 = 2 * filter_freq_end[i] * sinc(filter_freq_end[i] * freq_scale, t_right)
            band_pass = low_pass2 - low_pass1
            band_pass = band_pass / tf.reduce_max(band_pass)
            filters.append(band_pass * window)
        filters = tf.stack(filters)
        # filter_number (out_channels) * kernel_size (filter_width) --> [filter_width, in_channels, out_channels]
        filters = tf.transpose(filters)
        filters = tf.expand_dims(filters, 1)
        output = inputs
        output = tf.nn.conv1d(output, filters, stride=1, padding='SAME')
    return upper_encoder(output,
                         params,
                         is_training)
