import math
from tensorflow.compat.v1.nn import conv2d_backprop_input
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras 
import yaml
# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import tools


from tools.display import grid_display, heatmap_display
from tools.image import apply_grey_patch
from tools.saver import save_rgb

from tensorflow.python.ops        import gen_nn_ops
import copy
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state
from skimage.color import gray2rgb
from tqdm.auto import tqdm  
from tensorflow.keras                        import backend as K
from tools.lrp_utils                        import (get_model_params, 
                                          get_gammas, 
                                          get_heatmaps, infer_target_layer, 
                                          load_images,
                                          predict_labels, 
                                          visualize_heatmap)
from tools import lime_base
from skimage import segmentation
from tensorflow.keras.applications.vgg19 import VGG19 

# -------  Début Partie LRP    ------ #

# -------  Fin Partie LRP    ------ #
class LayerwiseRelevancePropagation:

  def __init__(self,img_dir, results_dir, model = None, model_name='vgg19', alpha=2, epsilon=1e-7):

    if model is None:
      model_name = model_name.lower()
      if model_name == 'vgg16':
        model_type = VGG16
      elif model_name == 'vgg19':
        model_type = VGG19
      else:
        raise 'Model name not one of VGG16 or VGG19'
        sys.exit()
      self.model = model_type(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
    else:
      self.model = model
    self.alpha = alpha
    self.beta = 1 - alpha
    self.epsilon = epsilon

    self.names, self.activations, self.weights = get_model_params(self.model)
    self.num_layers = len(self.names)

    self.relevance = self.compute_relevances()
    self.lrp_runner = K.function(inputs=[self.model.input, ], outputs=[self.relevance, ])

  def compute_relevances(self):
    r = self.model.output
    for i in range(self.num_layers-2, -1, -1):
      if 'fc' in self.names[i + 1]:
        r = self.backprop_fc(self.weights[i + 1][0], self.weights[i + 1][1], self.activations[i], r)
      elif 'flatten' in self.names[i + 1]:
        r = self.backprop_flatten(self.activations[i], r)
      elif 'pool' in self.names[i + 1]:
        r = self.backprop_max_pool2d(self.activations[i], r)
      elif 'conv' in self.names[i + 1]:
        r = self.backprop_conv2d(self.weights[i + 1][0], self.weights[i + 1][1], self.activations[i], r)
      else:
        raise 'Layer not recognized!'
        sys.exit()
    return r

  def backprop_fc(self, w, b, a, r):
    w_p = K.maximum(w, 0.)
    b_p = K.maximum(b, 0.)
    z_p = K.dot(a, w_p) + b_p + self.epsilon
    s_p = r / z_p
    c_p = K.dot(s_p, K.transpose(w_p))
    
    w_n = K.minimum(w, 0.)
    b_n = K.minimum(b, 0.)
    z_n = K.dot(a, w_n) + b_n - self.epsilon
    s_n = r / z_n
    c_n = K.dot(s_n, K.transpose(w_n))

    return a * (self.alpha * c_p + self.beta * c_n)

  def backprop_flatten(self, a, r):
    shape = a.get_shape().as_list()
    shape[0] = -1
    return K.reshape(r, shape)

  def backprop_max_pool2d(self, a, r, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1)):
    z = K.pool2d(a, pool_size=ksize[1:-1], strides=strides[1:-1], padding='valid', pool_mode='max')

    z_p = K.maximum(z, 0.) + self.epsilon
    s_p = r / z_p
    c_p = gen_nn_ops.max_pool_grad_v2(a, z_p, s_p, ksize, strides, padding='VALID')

    z_n = K.minimum(z, 0.) - self.epsilon
    s_n = r / z_n
    c_n = gen_nn_ops.max_pool_grad_v2(a, z_n, s_n, ksize, strides, padding='VALID')

    return a * (self.alpha * c_p + self.beta * c_n)

  def backprop_conv2d(self, w, b, a, r, strides=(1, 1, 1, 1)):
    w_p = K.maximum(w, 0.)
    b_p = K.maximum(b, 0.)
    z_p = K.conv2d(a, kernel=w_p, strides=strides[1:-1], padding='same') + b_p + self.epsilon
    s_p = r / z_p
    c_p = conv2d_backprop_input(K.shape(a), w_p, s_p, strides, padding='SAME')

    w_n = K.minimum(w, 0.)
    b_n = K.minimum(b, 0.)
    z_n = K.conv2d(a, kernel=w_n, strides=strides[1:-1], padding='same') + b_n - self.epsilon
    s_n = r / z_n
    c_n = conv2d_backprop_input(K.shape(a), w_n, s_n, strides, padding='SAME')

    return a * (self.alpha * c_p + self.beta * c_n)

  def predict_labels(self, images):
    return predict_labels(self.model, images)

  def run_lrp(self, images):
    print("Running LRP on {0} images...".format(len(images)))
    return self.lrp_runner([images, ])[0]

  def compute_heatmaps(self, images, g=0.2, cmap_type='rainbow', **kwargs):
    lrps = self.run_lrp(images)
    print("LRP run successfully...")
    gammas = get_gammas(lrps, g=g, **kwargs)
    print("Gamma Correction completed...")
    heatmaps = get_heatmaps(gammas, cmap_type=cmap_type, **kwargs)
    return heatmaps


# -------  Début Partie Grad-Cam    ------ #


class GradCam():

    def __init__(self, input_path, output_path, size) :
        self.input_path = input_path
        self.output_path = output_path
        self.size = size
        

    def get_img_array(img_path, size):
        # `img` is a PIL image of size 224x224
        img = keras.preprocessing.image.load_img(img_path, target_size=size)
        # `array` is a float32 Numpy array of shape (224, 224, 3)
        array = keras.preprocessing.image.img_to_array(img)
        # We add a dimension to transform our array into a "batch"
        # of size (1, 224, 224, 3)
        array = np.expand_dims(array, axis=0)
        return array


    def make_gradcam_heatmap(self, img_path, model, pred_index=None):
        
        img_array = self.get_img_array(img_path)

        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        last_conv_layer_name = list(filter(lambda x: isinstance(x, keras.layers.Conv2D), model.base.layers))[-1].name
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()


    def save_and_display_gradcam(self, img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
        # Load the original image
        img = keras.preprocessing.image.load_img(img_path)
        img = keras.preprocessing.image.img_to_array(img)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        # Save the superimposed image
        superimposed_img.save(cam_path)

        # Display Grad CAM
        display(Image(cam_path))

# -------  Fin Partie Grad-Cam    ------ #


# -------  Début Partie Occlusion sensitivity  ------ #

def explain(
        validation_data,
        model,
        class_index,
        patch_size,
        colormap=cv2.COLORMAP_VIRIDIS,
):
    """
    Compute Occlusion Sensitivity maps for a specific class index.

    Args:
        validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
            to perform the method on. Tuple containing (x, y).
        model (tf.keras.Model): tf.keras model to inspect
        class_index (int): Index of targeted class
        patch_size (int): Size of patch to apply on the image
        colormap (int): OpenCV Colormap to use for heatmap visualization

    Returns:
        np.ndarray: Grid of all the sensitivity maps with shape (batch_size, H, W, 3)
    """
    images, _ = validation_data
    sensitivity_maps = np.array(
        [
            get_sensitivity_map(model, image, class_index, patch_size)
            for image in images
        ]
    )

    heatmaps = np.array(
        [
            heatmap_display(heatmap, image, colormap)
            for heatmap, image in zip(sensitivity_maps, images)
        ]
    )

    grid = grid_display(heatmaps)

    return grid

def get_sensitivity_map(model, image, class_index, patch_size):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    """
    Compute sensitivity map on a given image for a specific class index.

    Args:
        model (tf.keras.Model): tf.keras model to inspect
        image:
        class_index (int): Index of targeted class
        patch_size (int): Size of patch to apply on the image

    Returns:
        np.ndarray: Sensitivity map with shape (H, W, 3)
    """
    batch_size = None # A changer si nécessaire

    sensitivity_map = np.zeros(
        (
            math.ceil(image.shape[0] / patch_size),
            math.ceil(image.shape[1] / patch_size),
        )
    )

    patches = [
        apply_grey_patch(image, top_left_x, top_left_y, patch_size)
        for index_x, top_left_x in enumerate(range(0, image.shape[0], patch_size))
        for index_y, top_left_y in enumerate(range(0, image.shape[1], patch_size))
    ]

    coordinates = [
        (index_y, index_x)
        for index_x in range(
            sensitivity_map.shape[1]  # pylint: disable=unsubscriptable-object
        )
        for index_y in range(
            sensitivity_map.shape[0]  # pylint: disable=unsubscriptable-object
        )
    ]

    predictions = model.predict(np.array(patches), batch_size=batch_size)
    print(predictions)
    input()
    target_class_predictions = [
        prediction[class_index] for prediction in predictions
    ]

    for (index_y, index_x), confidence in zip(
            coordinates, target_class_predictions
    ):
        sensitivity_map[index_y, index_x] = 1 - confidence

    return cv2.resize(sensitivity_map, image.shape[0:2])

def save(grid, output_dir, output_name):
    """
    Save the output to a specific dir.

    Args:
        grid (numpy.ndarray): Grid of all heatmaps
        output_dir (str): Output directory path
        output_name (str): Output name
    """
    save_rgb(grid, output_dir, output_name)

# -------  Fin Partie Occlusion sensitivity  ------ #





class ImageExplanation(object):
    def __init__(self, image, segments):
        """Init function.
        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}

    def get_image_and_mask(self, label, positive_only=True, negative_only=False, hide_rest=False,
                           num_features=5, min_weight=0.):
        """Init function.
        Args:
            label: label to explain
            positive_only: if True, only take superpixels that positively contribute to
                the prediction of the label.
            negative_only: if True, only take superpixels that negatively contribute to
                the prediction of the label. If false, and so is positive_only, then both
                negativey and positively contributions will be taken.
                Both can't be True at the same time
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of superpixels to include in explanation
            min_weight: minimum weight of the superpixels to include in explanation
        Returns:
            (image, mask), where image is a 3d numpy array and mask is a 2d
            numpy array that can be used with
            skimage.segmentation.mark_boundaries
        """
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_only & negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
        if positive_only:
            fs = [x[0] for x in exp
                  if x[1] > 0 and x[1] > min_weight][:num_features]
        if negative_only:
            fs = [x[0] for x in exp
                  if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
        if positive_only or negative_only:
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = -1 if w < 0 else 1
                temp[segments == f] = image[segments == f].copy()
                temp[segments == f, c] = np.max(image)
            return temp, mask



class LimeImageExplainer(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None):
        """Init function.
        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)

    def explain_instance(self, image, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=None, num_features=100000, num_samples=200,
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None,
                         progress_bar=True):
        """Generates explanations for a prediction.
        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).
        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: If not None, will hide superpixels with this color.
                Otherwise, use the mean pixel color of the image.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: batch size for model predictions
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.
            progress_bar: if True, show tqdm progress bar.
        Returns:
            An ImageExplanation object (see lime_image.py) with the corresponding
            explanations.
        """
        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        if segmentation_fn is None :
            segmentation_fn = segmentation.quickshift
        segments = segmentation_fn(image)

        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))
        else:
            fudged_image[:] = hide_color

        top = labels

        data, labels = self.data_labels(image, fudged_image, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size,
                                        progress_bar=progress_bar)

        distances = pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = ImageExplanation(image, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10,
                    progress_bar=True):
        """Generates images and predictions in the neighborhood of this image.
        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.
            progress_bar: if True, show tqdm progress bar.
        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, num_samples * n_features)\
            .reshape((num_samples, n_features))
        labels = []
        data[0, :] = 1
        imgs = []
        rows = tqdm(data) if progress_bar else data
        for row in rows:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)
        return data, np.array(labels)