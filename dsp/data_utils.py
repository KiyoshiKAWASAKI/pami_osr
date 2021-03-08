"""Data utilities for psychophysics oddball experiments using ImageNet."""
import csv
from dataclasses import dataclass
import json
import logging
import os

import numpy as np
import pandas as pd

from exputils.data import ConfusionMatrices


def save_img_ids(json_path, output_path, img_key='image'):
    """Obtains the img ids in the given JSON, saving it as a csv/txt file."""
    with open(json_path, 'r') as openf:
        sample_data = json.load(json_path)

    logging.info('Loaded `%s` with `%d` samples', json_path, len(sample_data))

    unique_imgs, counts = np.unique(
        [sample[img_key] for sample in sample_data],
        return_counts=True,
    )

    logging.info('`%d` unique images', len(unique_imgs))

    pd.DataFrame(
        np.hstack((unique_imgs, counts)),
        columns=['unique_image', 'sample_count'],
    ).to_csv(create_filepath(output_path), index=False)


def process_raw_csv(
    raw_csv_path,
    control_filepath,
    start_row=0,
    output_path=None,
    worker_id_col='AssignmentId',
    top_rt_percent=0.05,
    rm_top_rt=True,
    rm_control=True,
):
    """Process the raw CSV version of the Amazon Turk annotators, but preserve
    some information about the annotator, such as their performance on control
    questions.

    Args
    ----
    raw_csv_path : str
        Input filepath for the raw, unprocessed csv of experiment
        annotations
    control_filepath : str
        Filepath to the control questions, expecting YAML.
    start_row : int, optional
        The index of the row to start processing at, dropping all prior
        rows, e.g. drop the in-house testing results.
    output_path : str, optional
        Output filepath for the resulting processed csv
    worker_id_col : str, optional
    """
    raise NotImplementedError()

    # Load the raw csv
    raw_data = pd.read_csv(raw_csv_path)

    # Drop initial rows being skipped
    if start_row > 0:
        raw_data = raw_data.iloc[start_row:]

    logging.info('There are %d entries (rows) to be processed.', raw_data)

    # TODO save csv of annotator control question performance

    # TODO save "csv"/tensor of individual annotator confusion

    # TODO save csv of macro mean annotator class confusion
    # TODO save "csv"/tensor of macro mean annotator class RT
    #   stat summary: mean, sd, median, quantiles, min, max


    worker_ids = raw_data[worker_id_col].unique()

    # Get the control questions' image paths
    if os.path.splitext(control_filepath)[1] == 'npy':
        control_img_list = [
            question['image_paths'] for question in
            np.load(control_file_path, allow_pickle='TRUE').item()
        ]
    elif os.path.splitext(control_filepath)[1] == 'yaml':
        # TODO
        raise NotImplementedError()
    else:
        raise ValueError('Expected filetype of npy or yaml')

    # Get rid of the "[" and "]" in the image list
    raw_data['ImageList'] = raw_data['ImageList'].apply(lambda x: x[1:-1])

    control_scores = np.zeros([worker_ids.shape[0], len(control_img_list)])

    # Check the data for each worker
    for worker_id in worker_ids:
        # Get all the responses for one worker
        worker_response = raw_data.loc[
            raw_data[worker_id_col] == worker_id
        ]

        # Count the number of questions they answered
        nb_responses = worker_response.shape[0]

        # If this worker answered more than 27 questions, then we ignore his
        # responses.

        # The number of responses should be 25, but we allow 2 more entries
        if nb_responses >= 27 or nb_responses < 25:
            if nb_responses >= 27:
                logging.info(
                    'Annotator: `%s` removed for having 27 or more answers.'
                )
            elif nb_responses < 25:
                logging.info(
                    'Annotator: `%s` removed for having less than 25 answers.'
                )
            raw_data = raw_data.drop(
                raw_data[raw_data[worker_id_col] == worker_id].index
            )
            continue

        # Drop the rows from the workers who answred more than 2 control
        # questions wrong
        nb_control_wrong = 0

        for question in worker_response.itertuples(index=True, name="Pandas"):
            image_list = [x.strip() for x in question.ImageList.split(',')]

            image_list_formatted = []
            for image_path in image_list:
                image_list_formatted.append(image_path[1:-1])

            question_index = question.Index

            if image_list_formatted in control_img_list:
                if question.ImposterFound == 0:
                    nb_control_wrong += 1

                    # TODO here add which it is to update table

                # No matter what, drop this line of record (cause it is a
                # control question)
                raw_data = raw_data.drop(
                    raw_data[raw_data.index == question_index].index
                )

        if nb_control_wrong >= 3:
            # Dropping entries from a worker who answered 3 or more control
            # questions wrong.
            raw_data = raw_data.drop(
                raw_data[raw_data[worker_id_col] == worker_id].index
            )

    # Check the number of entries after processing and save it into csv file
    logging.info(
        "There are %d entries after processing the data.",
        raw_data.shape[0],
    )
    raw_data.reset_index(drop=True).to_csv(path_or_buf=processed_csv_save_path)


def annotator_control_question_score(
    csv_path,
    control_filepath,
    output_path,
    start_row=0,
    output_path=None,
    worker_id_col='AssignmentId',
):
    """Given the annotation csv and control questions, calculates each
    annotator's control question score, which is an integer between [0, 5].
    """
    # Load annotation csv
    data = pd.read_csv(csv_path)

    # Drop initial rows being skipped
    if start_row > 0:
        data = data.iloc[start_row:]

    logging.info('There are %d entries (rows) to be processed.', data)

    # TODO save csv of annotator control question performance
    worker_ids = raw_data[worker_id_col].unique()

    # TODO save "csv"/tensor of individual annotator confusion

    # TODO save csv of macro mean annotator class confusion
    # TODO save "csv"/tensor of macro mean annotator class RT
    #   stat summary: mean, sd, median, quantiles, min, max




    # Get rid of the "[" and "]" in the image list
    data['ImageList'] = data['ImageList'].apply(lambda x: x[1:-1])

    # TODO load control questions

    # TODO Get only the entries that are control questions

    # TODO Get the control question score of each unique annotator

    # TODO Save the resulting csv w/ header:
    #   worker_id_col, control_question_score

    raise NotImplementedError()


def annotator_confusion_matrices():
    csv_path,
    output_path,
    start_row=0,
    output_path=None,
    worker_id_col='AssignmentId',
):
    """Given the annotation csv and control questions, calculates each
    annotator's control question score, which is an integer between [0, 5].

    This should output to similar format as existing once modified, but also
    save the following:
        - Annotator Confusion Tensor with shape [Classes, Classes, Annotators,
          Ordering], where ordering is the class as known and class as unknown.
          Flattening on the ordering dim is probably the end result.
        - Annotator Reaction Time Tensor with shape [Classes, Classes,
          Annotators, Ordering].
        - Annotator Control Question Accuracy

    Ideally, all the above w/ annotators will order the annotators by their
    control question accuracy first, then date time by occurrence.

    """
    raise NotImplementedError()

    # TODO load csv

    # TODO Per unique annotator, get confusion matrix when class is known /
    # unknown or host / imposter.
    # TODO Save the resulting confusion tensor w/ the ordered labels:
    ConfusionMatrices(targets, preds, labels, dim_vars).save(output_path)
    # Technically it is still a tensor, but not a complete confusion tensor.
