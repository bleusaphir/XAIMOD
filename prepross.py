import cv2
import glob
import os
from PIL import Image, ImageEnhance
from PIL import ImageFilter
path1 = './Cancer_preprocessed/'
path2 = './Benign_preprocessed/'
if not os.path.exists('Cancer_preprocessed'):
    os.makedirs('Cancer_preprocessed/')
files1 = glob.glob('./Cancer_preprocessed/*')
for f in files1:
    os.remove(f)

if not os.path.exists('Benign_preprocessed'):
    os.makedirs('Benign_preprocessed/')
files2 = glob.glob('./Benign_preprocessed/*')
for f in files2:
    os.remove(f)


j=0
source_path1="./Cancer"
source_path2="./Benign"
inputShape=(224,224)
for file1, file2 in zip(os.listdir(source_path1), os.listdir(source_path2)):
    #print(file)
    image = Image.open(source_path1+"/"+file1)
    image = image.resize(inputShape, Image.ANTIALIAS)
    image = image.save(path1+'/'+ str(j) + ".png")
    j+=1
    z=0
    for filename in os.listdir(path1):
        import cv2

        # Loading .png image
        png_img = cv2.imread(path1+"/" + filename)

        # converting to jpg file
        # saving the jpg file
        cv2.imwrite(path1+'/' + str(z) + '.jpg', png_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        z += 1
    test = os.listdir(path1)
    for images in test:
        if images.endswith(".png"):
            os.remove(os.path.join(path1, images))
    
    image = Image.open(source_path2+"/"+file2)
    image = image.resize(inputShape, Image.ANTIALIAS)
    image = image.save(path2+'/'+ str(j) + ".png")
    j+=1
    z=0
    for filename in os.listdir(path2):
        import cv2

        # Loading .png image
        png_img = cv2.imread(path2+"/" + filename)

        # converting to jpg file
        # saving the jpg file
        cv2.imwrite(path2+'/' + str(z) + '.jpg', png_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        z += 1
    test = os.listdir(path2)
    for images in test:
        if images.endswith(".png"):
            os.remove(os.path.join(path2, images))
