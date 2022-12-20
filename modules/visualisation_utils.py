import imageio
import cv2
import os

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import pandas as pd

import numpy as np
import tensorflow as tf
from tensorflow_docs.vis import embed

import tensor_utils

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def _keypoints_and_edges_for_display(
        keypoints_with_scores,
        height,
        width,
        keypoint_threshold=0.11
):
    """
    Function done by TensorFlow team.

    Returns high confidence keypoints and edges for visualization.    

    Args:
      keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
      height: height of the image in pixels.
      width: width of the image in pixels.
      keypoint_threshold: minimum confidence score for a keypoint to be
        visualized.

    Returns:
      A (keypoints_xy, edges_xy, edge_colors) containing:
        * the coordinates of all keypoints of all detected entities;
        * the coordinates of all skeleton edges of all detected entities;
        * the colors in which the edges should be plotted.
    """

    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack(
            [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
        kpts_above_thresh_absolute = kpts_absolute_xy[
            kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)

        for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (kpts_scores[edge_pair[0]] > keypoint_threshold and
                    kpts_scores[edge_pair[1]] > keypoint_threshold):
                x_start = kpts_absolute_xy[edge_pair[0], 0]
                y_start = kpts_absolute_xy[edge_pair[0], 1]
                x_end = kpts_absolute_xy[edge_pair[1], 0]
                y_end = kpts_absolute_xy[edge_pair[1], 1]
                line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                keypoint_edges_all.append(line_seg)
                edge_colors.append(color)
    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
    else:
        keypoints_xy = np.zeros((0, 17, 2))

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2))
    return keypoints_xy, edges_xy, edge_colors


def draw_prediction_on_image(
        image,
        keypoints_with_scores,
        crop_region=None,
        close_figure=False,
        output_image_height=None
):
    """
    Function done by TensorFlow team.

    Draws the keypoint predictions on image.

    Args:
      image: A numpy array with shape [height, width, channel] representing the
        pixel values of the input image.
      keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
      crop_region: A dictionary that defines the coordinates of the bounding box
        of the crop region in normalized coordinates (see the init_crop_region
        function below for more detail). If provided, this function will also
        draw the bounding box on the image.
      output_image_height: An integer indicating the height of the output image.
        Note that the image aspect ratio will be the same as the input image.

    Returns:
      A numpy array with shape [out_height, out_width, channel] representing the
      image overlaid with keypoint predictions.
    """

    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))

    # To remove the huge white borders
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')

    im = ax.imshow(image)
    line_segments = LineCollection([], linewidths=(4), linestyle='solid')
    ax.add_collection(line_segments)
    # Turn off tick labels
    scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

    (keypoint_locs, keypoint_edges,
     edge_colors) = _keypoints_and_edges_for_display(
         keypoints_with_scores, height, width)

    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
    if keypoint_edges.shape[0]:
        line_segments.set_segments(keypoint_edges)
        line_segments.set_color(edge_colors)
    if keypoint_locs.shape[0]:
        scat.set_offsets(keypoint_locs)

    if crop_region is not None:
        xmin = max(crop_region['x_min'] * width, 0.0)
        ymin = max(crop_region['y_min'] * height, 0.0)
        rec_width = min(crop_region['x_max'], 0.99) * width - xmin
        rec_height = min(crop_region['y_max'], 0.99) * height - ymin
        rect = patches.Rectangle(
            (xmin, ymin), rec_width, rec_height,
            linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    if output_image_height is not None:
        output_image_width = int(output_image_height / height * width)
        image_from_plot = cv2.resize(
            image_from_plot, dsize=(output_image_width, output_image_height),
            interpolation=cv2.INTER_CUBIC)
    return image_from_plot


def _create_overlay(image_tensor, keypoints_with_scores):
    """
    Function done by TensorFlow team and modified by me.

    Creates overlay for image.

    Args:
        image_tensor (tensor): Image converted to tensor.
        keypoints_with_scores (numpy array): Array containing keypoints and confidence values.

    Returns:
        output_overlay (np array): Image with keypoints overlaid converted to Numpy array.

    """
    display_image = tf.expand_dims(image_tensor, axis=0)
    display_image = tf.cast(tf.image.resize_with_pad(
        display_image, 1280, 1280), dtype=tf.int32)
    output_overlay = draw_prediction_on_image(
        np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)
    return output_overlay


def visualize_image_with_keypoints(image_tensor, keypoints_with_scores):
    """
    Function done by TensorFlow team and modified by me.

    Displays single image with keypoints.

    Args:
        image_tensor (tensor): Image converted to tensor.
        keypoints_with_scores (numpy array): Array containing keypoints and confidence values.

    Returns
        None
    """

    output_overlay = _create_overlay(image_tensor, keypoints_with_scores)

    plt.figure(figsize=(5, 5))
    plt.imshow(output_overlay)
    _ = plt.axis('off')


def visualize_multiple_images(samples: list, model_names: list):
    """
    Function for displaying multiple images of samples provided by benchmark.

    Args:
        samples (list): List of lists containing model_name, image_path and keypoints_with_scores.
        model_names (list): List of model_names, used for row amount.

    Returns:
        None
    """

    fig = plt.figure(figsize=(40, 40))

    rows = len(model_names)

    columns = int(len(samples) / rows)

    fig.suptitle('Visual comparison of pose estimation models',
                 fontsize=10*columns)

    print(f"Creating {rows} x {columns} visualization collage")

    for i in range(0, columns*rows):
        print(f"- Handling image {i+1}/{columns*rows}", end="\r")
        model_name = samples[i][0]
        image_path = samples[i][1]
        image_tensor = tensor_utils.convert_image_to_tensor(image_path)
        keypoints_with_scores = samples[i][2]

        img = _create_overlay(image_tensor, keypoints_with_scores)
        ax = fig.add_subplot(rows, columns, i+1)

        if i % columns == 0:
            ax.set_ylabel(model_name, fontsize=20)
        if i < columns:
            ax.set_xlabel(os.path.basename(image_path), fontsize=20)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.xaxis.set_label_position('top')
        plt.imshow(img)

    print(f'{"- Composing..." : <30}')

    plt.show()


def visualize_box_plot(
        df: pd.DataFrame,
        grouping: str,
        to_plot: str,
        x_label: str,
        y_label: str,
        title: str):
    """
    Visualize wanted data as box plots.

    Args:
        df (Pandas dataframe): Input dataframe.
        grouping (str): Column to group results by.
        to_plot (str): Column which results to plot.
        x_label (str): Label of the x axis, should match 'grouping' column.
        y_label (str): Label of the y axis, should match 'to_plot' column.
        title (str): Title of the plot.

    Returns:
        None
    """

    data = []
    groups = df[grouping].unique().tolist()

    for model_name in groups:
        data_list = df[df[grouping] == model_name][to_plot].values.tolist()
        data.append(data_list)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    bp = ax.boxplot(data)

    plt.xticks(ticks=list(range(1, len(groups)+1)), labels=groups, rotation=10)
    plt.ylabel(y_label, labelpad=10)
    plt.xlabel(x_label, labelpad=10)
    plt.grid(axis="y")
    plt.title(title)

    plt.show()
