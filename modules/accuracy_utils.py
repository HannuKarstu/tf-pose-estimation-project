import os
import math
# import posenet

import numpy as np


MOVENET_TO_MPII_DICT = {
    # Keys: MoveNet - Values: MPII
    0: None,  # 'nose'
    1: None,  # 'left_eye'
    2: None,  # 'right_eye'
    3: None,  # 'left_ear'
    4: None,  # 'right_ear'
    5: 13,    # 'left_shoulder'
    6: 12,    # 'right_shoulder'
    7: 14,    # 'left_elbow'
    8: 11,    # 'right_elbow'
    9: 15,    # 'left_wrist'
    10: 10,   # 'right_wrist'
    11: 3,    # 'left_hip'
    12: 2,    # 'right_hip'
    13: 4,    # 'left_knee'
    14: 1,    # 'right_knee'
    15: 5,    # 'left_ankle'
    16: 0     # 'right_ankle'
}

MOVENET_TO_BLAZEPOSE_DICT = {
    # Keys: MoveNet - Values: BlazePose
    0: 0,     # 'nose'
    1: 2,     # 'left_eye'
    2: 5,     # 'right_eye'
    3: 7,     # 'left_ear'
    4: 8,     # 'right_ear'
    5: 11,    # 'left_shoulder'
    6: 12,    # 'right_shoulder'
    7: 13,    # 'left_elbow'
    8: 14,    # 'right_elbow'
    9: 15,    # 'left_wrist'
    10: 16,   # 'right_wrist'
    11: 23,   # 'left_hip'
    12: 24,   # 'right_hip'
    13: 25,   # 'left_knee'
    14: 26,   # 'right_knee'
    15: 27,   # 'left_ankle'
    16: 28    # 'right_ankle'
}


def get_matching_annotation_from_data(
    data: list,
    image_path: str
):
    """
    Find matching annotation for image.

    Args:
        data (list): List of annotations in dict format
        image_path (str): Path of the image

    Returns:
        Annotation which matches the image. If not found, returns None.

    """
    filename = os.path.basename(image_path)

    for item in data:
        if item.get("image") == filename:
            return item

    return None


def filenames_from_annotations(annotations: list):
    """
    Get filenames from annotations a list

    Args:
        annotations (list): List of dictionaries

    Returns:
        filenames (list): List of filenames
    """

    filenames = []

    for item in annotations:
        filenames.append(item.get("image"))

    return filenames


def calculate_accuracy(
    actual_keypoints: np.array,
    predicted_keypoints: np.array
):
    """
    Calculates accuracy of predicted keypoints by comparing them
      to actual keypoints.

    Args:
        actual_keypoints (numpy array): Array containing actual keypoints.
        predicted_keypoints (numpy array): Array containing predicted keypoints and confidence values.

    Returns:
        accuracy (float): Accuracy percentage between 0 and 1.
    """

    accuracies = []

    for i in range(0, len(actual_keypoints[0][0])):
        if actual_keypoints[0][0][i][0] > -0.1 and actual_keypoints[0][0][i][1] > -0.1:
            x_actual = actual_keypoints[0][0][i][1]
            y_actual = actual_keypoints[0][0][i][0]
            x_predicted = predicted_keypoints[0][0][i][1]
            y_predicted = predicted_keypoints[0][0][i][0]

            distance = np.sqrt((x_predicted - x_actual) **
                               2 + (y_predicted - y_actual)**2)
            kp_fault = distance / math.sqrt(2)
            if x_predicted < 0.0 or y_predicted < 0.0:
                kp_fault = 1
            accuracies.append(kp_fault)

    accuracy = 1 - (sum(accuracies) / len(accuracies))

    return accuracy


def convert_posenet_results_to_movenet_keypoints_with_scores(results: list):
    """
    Converts PoseNet's results to match MoveNet keypoints_with_scores format.

    Not working at the moment!

    Args:
        results (list of arrays): List of arrays containing results from PoseNet

    Returns:
        keypoints_with_scores [1, 1, 17, 3] (float numpy array): representing the predicted keypoint
          coordinates and scores.
    """

    [heatmaps_result, offsets_result, displacement_fwd_result,
        displacement_bwd_result] = results

    # pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
    #     heatmaps_result.squeeze(axis=0),
    #     offsets_result.squeeze(axis=0),
    #     displacement_fwd_result.squeeze(axis=0),
    #     displacement_bwd_result.squeeze(axis=0),
    #     output_stride=8,
    #     min_pose_score=0.25)

    # converted = []
    # keypoints_with_scores = np.array([[converted]])
    # return keypoints_with_scores


def convert_blazepose_results_to_movenet_keypoints_with_scores(results):
    """
    Convert BlazePose results to similar format as MoveNet's output keypoints_with_scores.

    Args:
        results (Mediapipe pose process): Object containing the keypoints 
          and scores for 33 keypoints.

    Returns:
        keypoints_with_scores [1, 1, 17, 3] (float numpy array): representing the predicted keypoint
          coordinates and scores.
    """

    converted = []

    for key, value in MOVENET_TO_BLAZEPOSE_DICT.items():
        # y = 0 top, 1 bottom
        # x = 0 left, 1 right

        if results and results.pose_landmarks:
            y = results.pose_landmarks.landmark[value].y
            x = results.pose_landmarks.landmark[value].x
            confidence = results.pose_landmarks.landmark[value].visibility
        else:
            y, x, confidence = -1.0, -1.0, 0.0

        converted.append([y, x, confidence])

    keypoints_with_scores = np.array([[converted]])

    return keypoints_with_scores


def convert_annotation_item_to_keypoints_with_scores(
    annotation: dict,
    img_size: dict,
    annotation_type: str = "mpii"
):
    """
    Convert image annotation dictionary to 'keypoints_with_scores' 
      numpy array similar to TensorFlow's output.

    Coordinates in the MPII annotations are in pixels. They have to normalized and the
      black bars above and below the image have to be taken into account. This function 
      does that.

    Args:
        annotation (dict): Annotation dictionary
        im_size (dict): Images x and y size in pixels
        annotation_type (str): Default "mpii". Support ready for images from multiple sources.

    Returns:
        Numpy array with coordinates and confidence values of matching keypoints.

    """
    keypoints_with_scores = []
    keypoints_with_scores.append([])
    keypoints_with_scores[0].append([])

    if annotation_type == "mpii":
        conversion_dict = MOVENET_TO_MPII_DICT

    for key, value in conversion_dict.items():
        if value:
            y, x = annotation["joints"][value][1], annotation["joints"][value][0]
        else:
            y, x = -1.0, -1.0

        if y == -1.0 or x == -1.0:
            confidence = 0.0
        else:
            confidence = 1.0
            y = (y + (img_size["x"] - img_size["y"])/2) / \
                img_size["x"]
            x = x/img_size["x"]

        keypoint = [y, x, confidence]

        keypoints_with_scores[0][0].append(keypoint)

    keypoints_with_scores = np.array(keypoints_with_scores)
    return keypoints_with_scores
