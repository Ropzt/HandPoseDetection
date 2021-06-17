"""
IMPORTS
"""

import tensorflow as tf
import numpy as np
import ProjectionLayer as pj


"""
def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
"""

def identity_block(X, f, filters, step) :
    """

    :param X:
    :param f:
    :param filters:
    :param step:
    :return:
    """
    # Retrieve Filters
    F1, F2 = filters

    # Save the input value. We'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = tf.keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name='Conv_'+str(step)+'_1a',
                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name='Batch_Conv'+str(step)+'_1a')(X)
    X = tf.keras.layers.Activation('relu', name='Relu_Conv_'+str(step)+'_1a')(X)

    # Second component of main path
    X = tf.keras.layers.Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1), padding='same', name='Conv_'+str(step)+'_2a',
                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name='Batch_Conv'+str(step)+'_2a')(X)
    X = tf.keras.layers.Activation('relu', name='Relu_Conv_'+str(step)+'_2a')(X)

    # Third component of main path
    X = tf.keras.layers.Conv2D(filters=F2, kernel_size=(1, 1), strides=(1, 1), padding='valid', name='Conv_'+str(step)+'_3a',
                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name='Batch_Conv'+str(step)+'_3a')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = tf.keras.layers.Add(name='Product_Conv_'+str(step)+'_3a')([X, X_shortcut])
    X = tf.keras.layers.Activation('relu', name='Relu_Conv_'+str(step)+'_3a')(X)

    return X


def convolutional_block(X, f, filters, step, s=2):
    """

    :param X:
    :param f:
    :param filters:
    :param step:
    :param s:
    :return:
    """
    # Retrieve Filters
    F1, F2 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = tf.keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name='Conv_'+str(step)+'_1a',
                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name='Batch_Conv'+str(step)+'_1a')(X)
    X = tf.keras.layers.Activation('relu', name='Relu_Conv_'+str(step)+'_1a')(X)

    # Second component of main path
    X = tf.keras.layers.Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1), padding='same', name='Conv_'+str(step)+'_2a',
                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name='Batch_Conv'+str(step)+'_2a')(X)
    X = tf.keras.layers.Activation('relu', name='Relu_Conv_'+str(step)+'_2a')(X)

    # Third component of main path
    X = tf.keras.layers.Conv2D(filters=F2, kernel_size=(1, 1), strides=(1, 1), padding='valid', name='Conv_'+str(step)+'_3a',
                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name='Batch_Conv'+str(step)+'_3a')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = tf.keras.layers.Conv2D(F2, (1, 1), strides=(s, s), padding='valid', name='Conv_'+str(step)+'_1b',
                                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = tf.keras.layers.Add(name='Product_Conv_'+str(step)+'_3a')([X, X_shortcut])
    X = tf.keras.layers.Activation('relu', name='Relu_Conv_'+str(step)+'_3a')(X)

    return X

class RegNet34() :

    def __init__(self,input_shape=(256, 256, 3), heatmap_shape=(32,32)):
        """

        :param input_shape:
        :param heatmap_shape:
        """
        step=1
        # Define the input as a tensor with shape input_shape
        X_input = tf.keras.layers.Input(input_shape)

        # Zero-Padding
        X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)

        # Stage 1
        X = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), name='Conv_'+str(step)+'_1a', padding='valid', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(X)
        X = tf.keras.layers.BatchNormalization(axis=3, name='Batch_Conv'+str(step)+'_1a')(X)
        X = tf.keras.layers.Activation('relu', name='Relu_Conv_'+str(step)+'_1a')(X)
        X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name='MaxPool_'+str(step)+'_1a')(X)

        # Stage 2
        step+=1
        X = convolutional_block(X, f=3,step=step, filters=[64, 256], s=1)
        step += 1
        X = identity_block(X, 3, [64, 256],step=step)
        step += 1
        X = identity_block(X, 3, [64, 256],step=step)

        # Stage 3
        step += 1
        X = convolutional_block(X, f=3, step=step,filters=[128, 512], s=2)
        step += 1
        X = identity_block(X, 3, [128, 512],step=step)
        step += 1
        X = identity_block(X, 3, [128, 512],step=step)

        # Stage 4
        step += 1
        X = convolutional_block(X, f=3, step=step,filters=[256, 1024], s=2)
        step += 1
        X = identity_block(X, 3, [256, 1024],step=step)
        step += 1
        X = identity_block(X, 3, [256, 1024],step=step)
        step += 1
        X = identity_block(X, 3, [256, 1024],step=step)

        # Stage 5
        step += 1
        X = tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', name='Conv_'+str(step)+'_1a',
                                   kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(X)
        X = tf.keras.layers.BatchNormalization(axis=3, name='Batch_Conv'+str(step)+'_1a')(X)
        X = tf.keras.layers.Activation('relu', name='Relu_Conv_'+str(step)+'_1a')(X)

        # Stage 6
        step += 1
        X = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='Conv_'+str(step)+'_1a',
                                   kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(X)
        X = tf.keras.layers.BatchNormalization(axis=3, name='Batch_Conv'+str(step)+'_1a')(X)
        X = tf.keras.layers.Activation('relu', name='Relu_Conv_'+str(step)+'_1a')(X)

        # Spliting
        X_2D = X


        # 2D Heatmaps Generation
        step += 1
        X_2D = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='Conv_'+str(step)+'_2D',
                                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(X_2D)
        X_2D = tf.keras.layers.Conv2DTranspose(21, (4, 4), strides=(2, 2), padding='same', name='DEConv_'+str(step)+'_2D_1a',
                                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(X_2D)
        X_2D = tf.keras.layers.Conv2DTranspose(21, (4, 4), strides=(2, 2), padding='same', name='DEConv_'+str(step)+'_2D_2a',
                                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(X_2D)

        # 3D prediction
        X_3D = tf.keras.layers.Flatten(name='Flatten_'+str(step)+'_1a')(X)
        X_3D = tf.keras.layers.Dense(200,name='Dense_'+str(step)+'_1a')(X_3D)
        X_3D = tf.keras.layers.Dense(63,name='Dense_'+str(step)+'_2a')(X_3D)
        X_3D = tf.keras.layers.Reshape((21, 1, 3),name='Reshape_'+str(step)+'_1a')(X_3D)
        temp = tf.keras.layers.Reshape((21, 3),name='Reshape_'+str(step)+'_2a')(X_3D)

        # Proj Layer
        projLayer = pj.ProjLayer(heatmap_shape, name='projlayer')(temp)
        heatmaps_pred3D = pj.RenderingLayer(heatmap_shape, coeff=1, name='renderinglayer')(projLayer)
        heatmaps_pred3D_reshape = pj.ReshapeChannelToLast(heatmap_shape, name='reshapelayer')(heatmaps_pred3D)

        # Rendering
        step += 1
        X_rendered = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', name='Conv_'+str(step)+'_rendering',
                                            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(heatmaps_pred3D_reshape)
        X_rendered = tf.keras.layers.BatchNormalization(axis=3, name='Batch_Conv'+str(step)+'_1a')(X_rendered)
        X_rendered = tf.keras.layers.Activation('relu', name='Relu_Conv_'+str(step)+'_1a')(X_rendered)
        step += 1
        X_rendered = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', name='Conv_'+str(step)+'_rendering',
                                            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(X_rendered)
        X_rendered = tf.keras.layers.BatchNormalization(axis=3, name='Batch_Conv'+str(step)+'_1a')(X_rendered)
        X_rendered = tf.keras.layers.Activation('relu', name='Relu_Conv_'+str(step)+'_1a')(X_rendered)
        step += 1
        X_concat = tf.keras.layers.concatenate([X, X_rendered],name='Concat_'+str(step)+'_1a')
        X_concat = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='Conv_'+str(step)+'_rendering',
                                          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(X_concat)
        X_concat = tf.keras.layers.BatchNormalization(axis=3, name='Batch_Conv'+str(step)+'_1a')(X_concat)
        X_concat = tf.keras.layers.Activation('relu', name='Relu_Conv_'+str(step)+'_1a')(X_concat)

        # Final Heatmap
        step += 1
        X_heatmap = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='Conv_'+str(step)+'_heatmap',
                                           kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(X_concat)
        X_heatmap = tf.keras.layers.Conv2DTranspose(21, (4, 4), strides=(2, 2), padding='same', name='DEConv_'+str(step)+'_heatmap_1a',
                                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(
            X_heatmap)
        X_heatmap = tf.keras.layers.Conv2DTranspose(21, (4, 4), strides=(2, 2), padding='same', name='DEConv_'+str(step)+'_heatmap_2a',
                                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(
            X_heatmap)

        # Final 3D Joints
        X_concat = tf.keras.layers.Flatten(name='Flatten_'+str(step)+'_1a')(X_concat)
        X_3Dj = tf.keras.layers.Dense(200,name='Dense_'+str(step)+'_1a')(X_concat)
        X_3Dj = tf.keras.layers.Dense(63,name='Dense_'+str(step)+'_2a')(X_3Dj)
        X_3Dj = tf.keras.layers.Reshape((21, 1, 3),name='Reshape_'+str(step)+'_1a')(X_3Dj)

        # Create model
        self.model = tf.keras.Model(inputs=X_input, outputs=[X_3D,X_3Dj,X_heatmap], name='RegNet34')


    def train_on_batch(self, epoch, generator) :
        """

        :param epoch:
        :param generator:
        :return:
        """
        for i in range(0, epoch):
            image, crop_param, joint_3d, joint_3d_rate, joint_2d = generator.generate_batch()
            joint_3d_rate = np.reshape(joint_3d_rate, (-1, 21, 1, 3))
            result = self.model.train_on_batch(x=[image], y=[joint_3d_rate, joint_3d_rate, joint_2d])
            self.test_on_batch(generator, i + 1)

    def test_on_batch(self, generator, epoch):
        """

        :param generator:
        :param epoch:
        :return:
        """
        min_loss = [10000.0, 10000., 100000., 100000., 100000., 100000., 100000.]
        sum_result = [0.0, 0., 0., 0., 0., 0., 0.]
        image, crop_param, joint_3d, joint_3d_rate, joint_2d = generator.generate_batch()
        joint_3d_rate = np.reshape(joint_3d_rate, (-1, 21, 1, 3))
        result = self.model.test_on_batch(x=[image], y=[joint_3d_rate, joint_3d_rate, joint_2d])
        result = np.asarray(result)
        sum_result = np.asarray(sum_result)
        sum_result += result
        idx = generator.batch_size
        sum_result /= idx
        if min_loss[0] > sum_result[0]:
            min_loss = sum_result
            print(epoch, min_loss)

            # Handling the name ambiguities in the metadata creating the h5 file.
            for i, w in enumerate(self.model.weights):
                new_name = w.name + str(np.random.randint(0, 10025))+str(np.random.randint(0, 10025))
                self.model.weights[i]._handle_name = new_name
            for i, var in enumerate(self.model.optimizer.weights):
                name = 'variable{}'.format(i)
                self.model.optimizer.weights[i] = tf.Variable(var, name=name)

            self.model.save('model.h5')
            self.model.save_weights('./weights/weights.h5')

            idx = (idx + 1)


