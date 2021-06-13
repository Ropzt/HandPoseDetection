"""
IMPORTS
"""

import tensorflow as tf
import numpy as np
from PIL import Image


def make_1024_list() :
  """
  Generates a list of 1024 strings of numbers in the format "XXXX", filled with zeros before the number.
  It is here to translate integers into the format used in the dataset (+1 because the list starts at "0001").

  :return: returns a list of 1024 strings
  """
  list = []
  for x in range(1,10) :
    list.append('000'+str(x))
  for x in range(10,100) :
    list.append('00'+str(x))
  for x in range(100,1000) :
    list.append('0'+str(x))
  for x in range(1000,1025) :
    list.append(str(x))
  return list

def multivariate_gaussian(pos, mu, Sigma):
  n = mu.shape[0]
  Sigma_det = np.linalg.det(Sigma)
  Sigma_inv = np.linalg.inv(Sigma)
  N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
  fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)
  return np.exp(-fac / 2) / N


def gaussian_heat_map(x, N):
  X = np.linspace(0, N, N)
  Y = np.linspace(0, N, N)
  X, Y = np.meshgrid(X, Y)
  mu = np.array([x[0], x[1]])
  Sigma = np.array([[3.0, 0.], [0., 3.]])
    # Pack X and Y into a single 3-dimensional array
  pos = np.empty(X.shape + (2,))
  pos[:, :, 0] = X
  pos[:, :, 1] = Y
  Z = multivariate_gaussian(pos, mu, Sigma)
  return Z

def extract_from_txt(path,  out_heatmap,  heatmap_shape=[32,32]) :
  """
  Extract content properly from text files.

  :param path:
  :param out_heatmap:
  :param heatmap_shape:
  :return:
  """
  target = open(path)
  target = np.genfromtxt(target, delimiter=',', dtype="uint8")
  target = np.asarray(target, np.float)
  if out_heatmap :
    joint_2d_heatmap = []
    target = np.reshape(target, (21, 2))
    for val in target:
      heat_map = gaussian_heat_map(val / 8, heatmap_shape[0])
      joint_2d_heatmap.append(heat_map)
    target = joint_2d_heatmap
  return target



"""
LOAD DATA
"""

class batch_maker(tf.keras.utils.Sequence) :

  def __init__(self, batch_size=20):
    self.batch_size = batch_size

  def generate_batch(self) :
    """
    :return: a batch_size quantity of 1 input and 4 outputs
    """
    num_list = make_1024_list()
    rand_list = np.random.randint(0,1025,int(self.batch_size/2))
    rand_folder = int(np.random.randint(0, 141, 1))
    image=[]
    target_2D=[]
    target_3D=[]
    target_3Drate=[]
    target_crop=[]
    heatmap_shape = [32, 32]
    retry = True
    # Error handling
    while retry:
        retry = False
        error = False
        for y in rand_list:
            try:
                img_path_nobj = 'D:/HandsDataset/FAKEHANDS/GANeratedHands_Release/data/noObject/' + num_list[
                    rand_folder] + '/' + num_list[y] + '_color_composed.png'
                image_nobj = Image.open(img_path_nobj)
                img_path_obj = 'D:/HandsDataset/FAKEHANDS/GANeratedHands_Release/data/withObject/' + num_list[
                    rand_folder] + '/' + num_list[y] + '_color_composed.png '
                image_obj = Image.open(img_path_obj)
            except FileNotFoundError:
                print('The batch_maker tried to use a file that dont exist. Trying a new one.')
                error = True
        if error:
            retry = True
            rand_list = np.random.randint(0, 1025, int(self.batch_size/ 2))
            rand_folder = int(np.random.randint(0, 141, 1))

    for y in rand_list :

        img_path_nobj = 'D:/HandsDataset/FAKEHANDS/GANeratedHands_Release/data/noObject/'+num_list[rand_folder]+'/'+num_list[y]+'_color_composed.png'
        image_nobj = Image.open(img_path_nobj)
        image_nobj = np.asarray(image_nobj, dtype="uint8")
        image_nobj = np.asarray(image_nobj, np.float)
        image_nobj = image_nobj / 255.0
        image.append(image_nobj)

        img_path_obj = 'D:/HandsDataset/FAKEHANDS/GANeratedHands_Release/data/withObject/'+num_list[rand_folder]+'/'+num_list[y]+'_color_composed.png '
        image_obj = Image.open(img_path_obj)
        image_obj = np.asarray(image_obj, dtype="uint8")
        image_obj = np.asarray(image_obj, np.float)
        image_obj = image_obj / 255.0
        image.append(image_obj)

        target_path_2D_nobj = 'D:/HandsDataset/FAKEHANDS/GANeratedHands_Release/data/noObject/'+num_list[rand_folder]+'/'+num_list[y]+'_joint2D.txt'
        target_2D_nobj= extract_from_txt(target_path_2D_nobj,  out_heatmap=True)
        target_2D.append(target_2D_nobj)

        target_path_2D_obj = 'D:/HandsDataset/FAKEHANDS/GANeratedHands_Release/data/withObject/'+num_list[rand_folder]+'/'+num_list[y]+'_joint2D.txt'
        target_2D_obj = extract_from_txt(target_path_2D_obj,  out_heatmap=True)
        target_2D.append(target_2D_obj)

        target_path_3Drate_nobj = 'D:/HandsDataset/FAKEHANDS/GANeratedHands_Release/data/noObject/' + num_list[rand_folder]+'/'+num_list[y] + '_joint_pos.txt'
        target_3Drate_nobj = extract_from_txt(target_path_3Drate_nobj,  out_heatmap=False)
        target_3Drate.append(target_3Drate_nobj)

        target_path_3Drate_obj = 'D:/HandsDataset/FAKEHANDS/GANeratedHands_Release/data/withObject/' + num_list[rand_folder]+'/'+num_list[y] + '_joint_pos.txt'
        target_3Drate_obj = extract_from_txt(target_path_3Drate_obj, out_heatmap=False)
        target_3Drate.append(target_3Drate_obj)

        target_path_3D_nobj = 'D:/HandsDataset/FAKEHANDS/GANeratedHands_Release/data/noObject/' + num_list[rand_folder]+'/'+num_list[y] + '_joint_pos_global.txt'
        target_3D_nobj = extract_from_txt(target_path_3D_nobj,  out_heatmap=False)
        target_3D.append(target_3D_nobj)

        target_path_3D_obj = 'D:/HandsDataset/FAKEHANDS/GANeratedHands_Release/data/withObject/' + num_list[rand_folder]+'/'+num_list[y] + '_joint_pos_global.txt'
        target_3D_obj = extract_from_txt(target_path_3D_obj,  out_heatmap=False)
        target_3D.append(target_3D_obj)

        target_path_crop_nobj = 'D:/HandsDataset/FAKEHANDS/GANeratedHands_Release/data/noObject/' + num_list[rand_folder]+'/'+num_list[y] + '_crop_params.txt'
        target_crop_nobj = extract_from_txt(target_path_crop_nobj,  out_heatmap=False)
        target_crop.append(target_crop_nobj)

        target_path_crop_obj = 'D:/HandsDataset/FAKEHANDS/GANeratedHands_Release/data/withObject/' + num_list[rand_folder]+'/'+num_list[y] + '_crop_params.txt'
        target_crop_obj = extract_from_txt(target_path_crop_obj, out_heatmap=False)
        target_crop.append(target_crop_obj)

    image = np.asarray(image)

    target_crop = np.asarray(target_crop)
    target_crop = np.reshape(target_crop, (-1, 1, 3))

    target_3D = np.asarray(target_3D)
    target_3D = np.reshape(target_3D, (-1,63))

    target_3Drate = np.asarray(target_3Drate)

    target_2D = np.asarray(target_2D)
    target_2D = np.reshape(target_2D, (-1, 21, heatmap_shape[0], heatmap_shape[1]))
    target_2D = np.moveaxis(target_2D, 1, 3)

    print('A batch has been generated.')

    return image, target_crop, target_3D, target_3Drate, target_2D