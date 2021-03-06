"""
This file has been borrowed from ___, only small changes, like the get_config, have been made to make it run properly.
___, whoever you are, thank you for your precious help.
"""
"""
IMPORTS
"""

from keras.layers import *
import keras.backend as k_b
import numpy as np

class ProjLayer(Layer):
    def __init__(self, heatmap_shape, **kwargs):
        self.range = 1.5
        self.heatmap_size = heatmap_shape
        super(ProjLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ProjLayer, self).build(input_shape)

    def call(self,x,**kwargs):
        return (x[:,:,:2] + self.range) / (2*self.range) * (self.heatmap_size[0]-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 2)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'range': self.range,
            'heatmap_size': self.heatmap_size
        })
        return config


# 2d point [-1,1] to rendering gaussian 2D heat_map
class RenderingLayer(Layer):
    def __init__(self, output_shape, coeff, **kwargs):
        self.output_size = output_shape
        self.coeff = coeff
        self.base = k_b.ones((1, self.calc_cell_units()), dtype=np.float)
        self.ones = k_b.ones((21, 1, 2))
        self.board_ones = k_b.ones((21, self.calc_cell_units(), 2))

        pair = []
        for i in range(0, self.calc_cell_units()):
            pair.append((i%self.output_size[0], i//self.output_size[1]))
        pair = np.asarray(pair)

        self.back_board = k_b.ones((self.calc_cell_units(),2))
        k_b.set_value(self.back_board, pair)
        super(RenderingLayer, self).__init__(**kwargs)

    def calc_cell_units(self):
        return self.output_size[0]*self.output_size[1]

    def build(self, input_shape):
        super(RenderingLayer, self).build(input_shape)

    def call(self, x,**kwargs):
        joint_2d = x
        joint_2d = k_b.reshape(joint_2d, [-1, 21, 2])                # -1, 21, 1, 2

        joint_2d = k_b.reshape(joint_2d, [-1, 21, 1, 2])                # -1, 21, 1, 2
        joint_2d_ones = joint_2d * self.board_ones

        diff = (joint_2d_ones - self.back_board)
        fac = (k_b.square(diff[:, :, :, 0]) + k_b.square(diff[:, :, :, 1])) / (self.coeff)
        son_value = k_b.exp(-fac/2.0)
        mom_value = (2.0*np.pi) * (self.coeff)

        result = son_value / mom_value

        result = k_b.reshape(result, [-1, 21, self.output_size[0] * self.output_size[1]])
        return result

    def compute_output_shape(self, input_shape):
        input_a = input_shape
        return (input_a[0], 21, self.output_size[0], self.output_size[1])

    def get_config(self):
        config = super().get_config().copy()
        base_l = np.asarray(self.base)
        base_l = base_l.tolist()
        ones_l = np.asarray(self.ones)
        ones_l = ones_l.tolist()
        board_ones_l = np.asarray(self.board_ones)
        board_ones_l = board_ones_l.tolist()
        back_board_l = np.asarray(self.back_board)
        back_board_l = back_board_l.tolist()
        config.update({
            'output_size': self.output_size,
            'coeff': self.coeff,
            'base': base_l,
            'ones': ones_l,
            'board_ones': board_ones_l,
            'back_board': back_board_l
            })
        return config


class ReshapeChannelToLast(Layer):
    def __init__(self, heatmap_shape, **kwargs):
        self.base = k_b.ones((1, heatmap_shape[0]*heatmap_shape[1]), dtype=np.float)
        self.heatmap_shape = heatmap_shape
        super(ReshapeChannelToLast, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReshapeChannelToLast, self).build(input_shape)

    def call(self,x,**kwargs):
        x = k_b.reshape(x, (-1, 21, self.heatmap_shape[0]*self.heatmap_shape[1]))
        base = k_b.reshape(x[:,0,:] * self.base, (-1,self.heatmap_shape[0],self.heatmap_shape[1],1))
        for i in range(1, 21):
            test = (x[:,i,:] * self.base)
            test = k_b.reshape(test, (-1, self.heatmap_shape[0],self.heatmap_shape[1],1))
            base = k_b.concatenate([base,test])
        return base

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2], input_shape[3], input_shape[1])

    def get_config(self):
        config = super().get_config().copy()
        base_l = np.asarray(self.base)
        base_l = base_l.tolist()
        config.update({
            'base_b': base_l,
            'heatmap_shape': self.heatmap_shape
            })
        return config

