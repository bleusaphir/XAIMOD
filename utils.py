import math
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
from lrp import RelevancePropagation
import tools
tf.compat.v1.disable_eager_execution()

from tools.display import grid_display, heatmap_display
from tools.image import apply_grey_patch
from tools.saver import save_rgb

# -------  Début Partie LRP    ------ #

def plot_relevance_map(image, relevance_map, res_dir, i):
    """Plots original image next to corresponding relevance map.

    Args:
        image: original image
        relevance_map: relevance map of original image
        res_dir: path to directory where results are stored
        i: counter
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4, 2))
    image = image.eval(session=tf.compat.v1.Session())
    axes[0].imshow(image / 255.0)
    axes[1].imshow(relevance_map, cmap="afmhot")
    for ax in axes:
        ax.set_axis_off()
    plt.tight_layout()
    file_path = "{}{}{}".format(res_dir, i, ".png")
    plt.savefig(file_path, dpi=120)
    plt.close(fig)


def layer_wise_relevance_propagation(conf=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Bug avec CUDA : tf-GPU marche pas pour l'instant
    if conf is None:
        conf = yaml.safe_load(open("config.yml"))
    img_dir = conf["paths"]["image_dir"]
    res_dir = conf["paths"]["results_dir"]

    image_height = conf["image"]["height"]
    image_width = conf["image"]["width"]

    lrp = RelevancePropagation(conf)

    image_paths = list()
    for (dirpath, dirnames, filenames) in os.walk(img_dir):
        image_paths += [os.path.join(dirpath, file) for file in filenames]
    import PIL.Image
    for i, image_path in enumerate(image_paths):
        print("Processing image {}".format(i + 1))
        image = center_crop(np.array(PIL.Image.open(image_path).convert('RGB')), image_height, image_width)
        relevance_map = lrp.run(image)
        plot_relevance_map(image, relevance_map, res_dir, i)


def center_crop(image, image_height, image_width):
    """Crops largest central region of image.

    Args:
        image: array of shape (W, H, C)
        image_height: target height of image
        image_width: target width of image

    Returns:
        Cropped image

    Raises:
        Error if image is not of type RGB.
    """

    if (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[-1] == 1):
        raise Exception("Error: Image must be of type RGB.")

    h, w = image.shape[0], image.shape[1]

    if h > w:
        cropped_image = tf.image.crop_to_bounding_box(image, (h - w) // 2, 0, w, w)
    else:
        cropped_image = tf.image.crop_to_bounding_box(image, 0, (w - h) // 2, h, h)

    return tf.image.resize(cropped_image, (image_width, image_height))


# -------  Fin Partie LRP    ------ #


# -------  Début Partie Grad-Cam    ------ #

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
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


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
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