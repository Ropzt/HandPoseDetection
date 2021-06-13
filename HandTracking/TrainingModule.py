"""
IMPORTS
"""
import tensorflow as tf
import DataLoading as dl
import RegNet as rg

"""
MAIN
"""

gen = dl.batch_maker(batch_size=28)

net = rg.RegNet34()

optimizer = tf.keras.optimizers.Adadelta(lr=1e-4)

net.model.compile(optimizer=optimizer,
                         loss=['mse','mse','mse'],
                         loss_weights=[100, 100, 1],
                         metrics=['mse'])

net.train_on_batch(epoch=10000, generator=gen)
