import time
import os
import pandas as pd

import visualisation_utils
import model_utils
import tensor_utils
import file_utils
import accuracy_utils

dir_path = os.path.dirname(os.path.realpath(__file__))
BENCHMARK_DATAFRAMES_DIR = os.path.join(dir_path, "..", "benchmark_dataframes")


def open_dataframe_from_disk(
        filename: str,
        folder: str = BENCHMARK_DATAFRAMES_DIR):
    """
    Open pickled Pandas dataframe from disk.

    Args:
        filename (str): Name of .pkl file to open.

    Returns:
        Pandas dataframe
    """
    filepath = os.path.join(folder, filename)

    if not os.path.isfile(filepath):
        raise ValueError(f"No file found in {filepath}")

    df = pd.read_pickle(filepath)
    print(f"Opened pickled dataframe from '{filepath}'")

    return df


def save_dataframe_to_disk(
        filename: str,
        df: pd.DataFrame,
        folder: str = BENCHMARK_DATAFRAMES_DIR):
    """
    Save dataframe as pickle file to disk.

    Args:
        filename (str): Name of .pkl file to save
        df (Pandas dataframe): Dataframe to save to disk.
        folder (str): Default BENCHMARK_DATAFRAMES_DIR. Folder to save the .pkl file to.

    Returns:
        Filepath of pickled dataframe
    """

    if not filename.endswith(".pkl"):
        raise Exception(f"Filename has to be '.pkl' file: {filename}")

    filepath = os.path.join(folder, filename)
    print(f"Saving created dataframe to disk: '{filepath}'")
    df.to_pickle(filepath)

    if not os.path.isfile(filepath):
        raise ValueError(f"No file found in {filepath}")

    return filepath


def run_benchmark(
    model_names: list,
    images_folder: str,
    image_limit: int = None,
    save_result_df_to_disk: bool = False,
    result_df_filename: str = None,
    amount_of_sample_images_to_save: int = 0,
    annotations_file_path=None
):
    """
    Run benchmark

    Args:
        model_names (list): List of model names to benchmark.
        images_folder (str): Folder from which to get images.
        image_limit (int): Default None. Limit the amount of selected images.
        save_to_disk (bool): Default False. Whether to save the output dataframe to disk as pickle file.
        filename (str): Default None. Filename of the pickle file for saving the output dataframe.
        amount_of_sample_images_to_save (int): Default 0. Amount of sample images to save for visual comparison of keypoint predictions.
        annotation_file_path (str): Default None. Path to annotations file for accuracy calculation.

    Returns:
        df (Pandas dataframe)
        samples (list): List of sample images and their predicted keypoints.
    """
    print("Running benchmark\n")

    results = []
    samples = []
    filenames = None

    if annotations_file_path:
        annotations = file_utils.open_json_as_dict(annotations_file_path)
        # Use only images which have annotations
        filenames = accuracy_utils.filenames_from_annotations(annotations)

    extensions = ('.jpg', '.jpeg', '.png')
    image_paths = file_utils.filepaths_from_path(
        images_folder, extensions, image_limit, shuffle=True, filenames=filenames)

    for model_index, model_name in enumerate(model_names):
        print(
            f"Using model {model_index+1}/{len(model_names)} - '{model_name}'")

        keypoint_detector, input_size = model_utils.select_model(
            model_name)

        for image_index, image_path in enumerate(image_paths):
            print(
                f"- Evaluating image {image_index+1}/{len(image_paths)}", end="\r")

            image_tensor = tensor_utils.convert_image_to_tensor(
                image_path)

            start = time.perf_counter()

            if "movenet" in model_name:
                resized_image_tensor = tensor_utils.resize_image_tensor(
                    image_tensor, input_size)
                keypoints_with_scores = keypoint_detector(resized_image_tensor)
            elif "blazepose" in model_name:
                blazepose_results = keypoint_detector(image_path)
                keypoints_with_scores = accuracy_utils.convert_blazepose_results_to_movenet_keypoints_with_scores(
                    blazepose_results)

            completion_time = time.perf_counter() - start

            result = [
                model_name,
                completion_time,
                os.path.basename(image_path)
            ]

            if annotations:
                img_size = {
                    "x": image_tensor.shape[1],
                    "y": image_tensor.shape[0]
                }
                annotation = accuracy_utils.get_matching_annotation_from_data(
                    annotations, image_path)
                actual_keypoints = accuracy_utils.convert_annotation_item_to_keypoints_with_scores(
                    annotation, img_size)
                accuracy = accuracy_utils.calculate_accuracy(
                    actual_keypoints, keypoints_with_scores)
                result.append(accuracy)

            # Append keypoint confidence scores
            for item in enumerate(keypoints_with_scores[0][0]):
                result.append(item[1][2])

            results.append(result)

            # Save sample images
            if image_index < amount_of_sample_images_to_save:
                samples.append([model_name, image_path, keypoints_with_scores])

        print(f'{"- Done" : <30}')

    columns = ["model name", "completion time", "image name"]
    if annotations:
        columns.append("accuracy")
    columns.extend(list(visualisation_utils.KEYPOINT_DICT.keys()))
    df = pd.DataFrame(results, columns=columns)

    print(
        f"Benchmark finished. Evaluated {len(model_names)} models with {len(image_paths)} images.")
    print(f"Saved sampling data from {len(samples)} images.")

    if save_result_df_to_disk and result_df_filename:
        save_dataframe_to_disk(result_df_filename, df)

    return df, samples
