from tkinter import Tk
from tkinter.filedialog import askdirectory
import tensorflow as tf
import tensorflow.keras as keras
import utils
import os
# print("Welcome. Select your Folder : ")
# dataFolder = askdirectory(title='Dossier de donnees') # shows dialog box and return the path
# input(f"Dossier de donnees : {dataFolder}?")  

# destinationFolder = askdirectory(title='Dossier de destination')
# input(f"Dossier de destination : {destinationFolder}")
model = tf.keras.applications.vgg16.VGG16(weights="imagenet", include_top=True)

#utils.layer_wise_relevance_propagation()  #Test LRP

def testOcc(model):
    # Load the original image
    img = keras.preprocessing.image.load_img(r'C:\Users\jyann\XAIMOD\src\data\A_1329_1.LEFT_CC.png')
    img = keras.preprocessing.image.img_to_array(img)
    patch_size = 4
    output = utils.get_sensitivity_map(
        model=model, image=img, class_index=0, patch_size=patch_size
    )

    utils.save(output, output_dir="results/", output_name='test1.png')

testOcc(model)
#%%
