"""Data utilities for psychophysics oddball experiments using ImageNet."""
import csv
from dataclasses import dataclass
import json
import logging
import os

import numpy as np
import pandas as pd

#from exputils.data import ConfusionMatrices
from exputils.io import create_filepath


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
    output_path,
    annotator_output_path,
    start_row=0,
    worker_id_col='AssignmentId',
    rt_key='ResponseTime',
    top_rt_percent=0.05,
    rm_top_rt=False,
    rm_control=False,
    rm_lt_control=3,
    record_response_counts=None,
    low_response_thresh=25,
    high_response_thresh=27,
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
    # Load the raw csv
    raw_data = pd.read_csv(raw_csv_path)

    # Drop initial rows being skipped
    if start_row > 0:
        raw_data = raw_data.iloc[start_row:]

    logging.info('There are %d entries (rows) to be processed.', len(raw_data))

    worker_ids = raw_data[worker_id_col].unique()

    logging.info(
        'There are %d unique annotators prior to processing.',
        len(worker_ids),
    )

    # Get the control questions' image paths
    if os.path.splitext(control_filepath)[1] == '.npy':
        control_img_list = [
            question['image_paths'] for question in
            np.load(control_filepath, allow_pickle=True).item().values()
        ]
    elif os.path.splitext(control_filepath)[1] == '.yaml':
        # TODO
        raise NotImplementedError()
    else:
        raise ValueError('Expected filetype of npy or yaml')

    # Get rid of the "[" and "]" in the image list
    raw_data['ImageList'] = raw_data['ImageList'].apply(lambda x: x[1:-1])

    #Find top percent and either mark em or remove em.
    sorted_raw = raw_data.sort_values(rt_key)

    # Remove any negative reaction times.
    first_pos_idx = -1
    for i, row in enumerate(sorted_raw[rt_key]):
        if row >= 0:
            first_pos_idx = i
            break
    if first_pos_idx == -1:
        raise ValueError(
            f'No positive reaction times under rt_key = `{rt_key}`!',
        )
    elif first_pos_idx != 0:
        sorted_raw = sorted_raw.iloc[first_pos_idx:]

    # Mark the questions whose samples are in the top percent
    raw_data['in_rt_top_percent'] = False
    raw_data['in_rt_top_percent'][
        sorted_raw[rt_key].iloc[
            :-np.floor(len(sorted_raw) * top_rt_percent).astype(int)
        ].index
    ] = True


    if rm_top_rt:
        # Remove the questions whose samples are in the top percent
        raw_data.drop(sorted_raw['in_rt_top_percent'], inplace=True)


    # Create placeholder df for unique annotator control scores etc...
    num_control_qs = len(control_img_list)
    annotator_df = pd.DataFrame(
        np.full([worker_ids.shape[0], num_control_qs + 2], False),
        columns=
            [f'control_q{i + 1}' for i in range(num_control_qs)]
            + ['total_responses', 'in_rt_top_percent']
        ,
        index=worker_ids,
    )
    annotator_df.rename_axis(worker_ids)
    annotator_df['total_responses'] = 0
    annotator_df['in_rt_top_percent'] = raw_data['in_rt_top_percent']

    if record_response_counts:
        response_counts = annotator_df[[
            'total_responses',
            'in_rt_top_percent',
        ]].copy()
        response_counts.rename_axis(worker_ids)

    # TODO parallelize the following:
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
        if record_response_counts:
            response_counts.at[worker_id, 'total_responses'] = nb_responses
            response_counts.at[worker_id, 'in_rt_top_percent'] = \
                worker_response['in_rt_top_percent'].any()

        # The number of responses should be 25, but we allow 2 more entries
        if nb_responses >= high_response_thresh:
            logging.info(
                'Annotator: `%s` removed for having %d or more answers: %d',
                worker_id,
                high_response_thresh,
                nb_responses,
            )

            raw_data.drop(
                raw_data[raw_data[worker_id_col] == worker_id].index,
                inplace=True,
            )
            annotator_df.drop(worker_id, inplace=True)
            continue
        elif nb_responses < low_response_thresh:
            logging.info(
                'Annotator: `%s` removed for having less than %d answers: %d',
                worker_id,
                low_response_thresh,
                nb_responses,
            )

            raw_data.drop(
                raw_data[raw_data[worker_id_col] == worker_id].index,
                inplace=True,
            )
            annotator_df.drop(worker_id, inplace=True)
            continue

        annotator_df.at[worker_id, 'total_responses'] = nb_responses
        annotator_df.at[worker_id, 'in_rt_top_percent'] = \
            worker_response['in_rt_top_percent'].any()

        # Drop the rows from the workers who answred more than 2 control
        # questions wrong
        for question in worker_response.itertuples(index=True, name="Pandas"):
            image_list = [x.strip() for x in question.ImageList.split(',')]

            image_list_formatted = []
            for image_path in image_list:
                image_list_formatted.append(image_path[1:-1])

            question_index = question.Index

            if image_list_formatted in control_img_list:
                idx = control_img_list.index(image_list_formatted)

                if question.ImposterFound == 1:
                    annotator_df.at[worker_id, f'control_q{idx + 1}'] = True

                if rm_control:
                    # No matter what, drop this line of record (cause it is a
                    # control question)
                    raw_data.drop(
                        raw_data[raw_data.index == question_index].index,
                        inplace=True,
                    )

        control_score = annotator_df.loc[worker_id].iloc[:num_control_qs].sum()
        if control_score == 0:
            raw_data.drop(
                raw_data[raw_data[worker_id_col] == worker_id].index,
                inplace=True,
            )
            annotator_df.drop(worker_id, inplace=True)
        elif control_score < rm_lt_control:
            # Dropping entries from a worker who answered 3 or more control
            # questions wrong.
            raw_data.drop(
                raw_data[raw_data[worker_id_col] == worker_id].index,
                inplace=True,
            )

    # Check the number of entries after processing and save it into csv file
    logging.info(
        "There are %d entries after processing the data.",
        raw_data.shape[0],
    )

    logging.info(
        "There are %d unique annotators after processing the data.",
        annotator_df.shape[0],
    )

    raw_data.to_csv(create_filepath(output_path), index=False)
    annotator_df.to_csv(create_filepath(annotator_output_path))

    if record_response_counts:
        response_counts.to_csv(create_filepath(record_response_counts))


def annotator_confusion_matrices(
    csv_path,
    output_path,
    start_row=0,
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
    #ConfusionMatrices(targets, preds, labels, dim_vars).save(output_path)
    # Technically it is still a tensor, but not a complete confusion tensor.
