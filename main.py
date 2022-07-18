from tkinter import Tk
from tkinter.filedialog import askdirectory
import tensorflow as tf
import tensorflow.keras as keras
import utils
import os
import PIL
from tf_explain.core import GradCAM, OcclusionSensitivity
import matplotlib.pyplot as plt
import numpy as np
# print("Welcome. Select your Folder : ")
# dataFolder = askdirectory(title='Dossier de donnees') # shows dialog box and return the path
# input(f"Dossier de donnees : {dataFolder}?")  

# destinationFolder = askdirectory(title='Dossier de destination')
# input(f"Dossier de destination : {destinationFolder}")
model = keras.models.load_model('models/Last.h5')

#utils.layer_wise_relevance_propagation()  #Test LRP
print("Yo j'suis là")
def testOcc(model):
    # Load the original image
    img = keras.preprocessing.image.load_img('test/cancer/0.jpg')
    img = keras.preprocessing.image.img_to_array(img)

    patch_size = 4
    explainer = OcclusionSensitivity()
    output = explainer.explain(([img], None), patch_size=patch_size, model=model, class_index=1)

    utils.save(output, output_dir="results/", output_name='test1.png')


testOcc(model)




def testLime(model):
  from skimage.segmentation import mark_boundaries
  explainer = utils.LimeImageExplainer()
  img = PIL.Image.open('test/cancer/6.jpg')
  img = np.array(img)
  ret = explainer.explain_instance(img, model)
  temp, mask = ret.get_image_and_mask(1, positive_only=False, num_features=10, hide_rest=False, min_weight=0.)
  plt.imshow(mark_boundaries(temp, mask))
  

#%%
#testLime(model)