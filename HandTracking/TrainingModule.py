"""
IMPORTS
"""
import tensorflow as tf
import DataLoading as dl
import RegNet as rg

"""
MAIN
"""
# Enter the path to the /data folder in the GANeratedHands dataset you downloaded from https://handtracker.mpi-inf.mpg.de/projects/GANeratedHands/GANeratedDataset.htm
gen = dl.batch_maker(root='D:/HandsDataset/FAKEHANDS/GANeratedHands_Release',batch_size=28)

net = rg.RegNet34()

optimizer = tf.keras.optimizers.Adadelta(lr=1e-4)

net.model.compile(optimizer=optimizer,
                         loss=['mse', 'mse', 'mse'],
                         loss_weights=[100, 100, 1],
                         metrics=['mse'])

net.train_on_batch(epoch=10000, generator=gen)
