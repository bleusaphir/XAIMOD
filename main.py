from tkinter import Tk
from tkinter.filedialog import askdirectory
import tensorflow as tf
import tensorflow.keras as keras
import utils
import os
#from google.colab.patches import cv2_imshow
import PIL
from tf_explain.core import GradCAM, OcclusionSensitivity
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import *
from tensorflow.keras.applications.vgg19 import VGG19 
# print("Welcome. Select your Folder : ")
# dataFolder = askdirectory(title='Dossier de donnees') # shows dialog box and return the path
# input(f"Dossier de donnees : {dataFolder}?")  

# destinationFolder = askdirectory(title='Dossier de destination')
# input(f"Dossier de destination : {destinationFolder}")
model=VGG19(include_top=False,
                     weights='imagenet', 
                     input_shape=(224,224,3))

#utils.layer_wise_relevance_propagation()  #Test LRP
print("Yo j'suis là")
def testOcc(model):
    # Load the original image
    img = keras.preprocessing.image.load_img('test/cancer/1.jpg')
    img = keras.preprocessing.image.img_to_array(img)

    patch_size = 4
    explainer = OcclusionSensitivity()
    output = explainer.explain(([img], None), patch_size=patch_size, model=model, class_index=1)

    #utils.save(output, output_dir="results/", output_name='test1.png')
    #cv2_imshow(output)


#testOcc(model)

def testLRP(model):
  from tensorflow.python.framework.ops import disable_eager_execution
  disable_eager_execution()
  from tools.lrp_utils import load_images, visualize_heatmap
  img_dir = './test/cancer/'
  results_dir = './results/'
  imgs_names = os.listdir(img_dir) 

  image_paths = [img_dir + name for name in imgs_names]
  image_names = [name.split('.')[0] for name in imgs_names]

  num_images = len(image_names)
  raw_images, processed_images = load_images(image_paths)
  explainer = utils.LayerwiseRelevancePropagation(img_dir, results_dir)
  labels = explainer.predict_labels(processed_images)
  print("Labels predicted...")
  heatmaps = explainer.compute_heatmaps(processed_images)
  print("Heatmaps generated...")

  for img, hmap, label, name in zip(raw_images, heatmaps, labels, image_names):
    visualize_heatmap(img, hmap, label, results_dir + name + '.jpg')

testLRP(model)


def testLime(model):
  from skimage.segmentation import mark_boundaries
  explainer = utils.LimeImageExplainer(verbose = True)
  img = PIL.Image.open('test/cancer/11.jpg')
  img = np.array(img)
  ret = explainer.explain_instance(img, model, num_features = 100000, num_samples = 2500)
  temp, mask = ret.get_image_and_mask(1, positive_only=False, num_features=10, hide_rest=False)
  plt.imshow(mark_boundaries(temp, mask))
  
  

#%%
#testLime(model)