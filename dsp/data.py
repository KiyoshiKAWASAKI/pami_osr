"""Data utilities for psychophysics oddball experiments using ImageNet."""
import csv
from dataclasses import dataclass
import json
import logging
import os

import torch

from exputils.data import ConfusionMatrices

@dataclass
class ContinuousSummaryStats:
    mean: float
    max: float
    min: float
    median: float = None
    quantiles: float = None
    num_samples: int = None

@dataclass
class PsychophysicsLabel:
    """Class for managing label information per image sample."""
    label: object
    reaction_time: ContinuousSummaryStats # instance reaction time
    class_reaction_time: ContinuousSummaryStats # Label Class reaction time
    #prob_vector: np.ndarray # possibly should be a 1d Torch tensor
    # pairwise_RT, perhaps class_reaction_time can contain this? or this could
    # be accessed by using the label index.

class PsychophysicsStats(object):
    """The psychophysics statistics and information."""

    # Annotator Confusion Matrix

    # Annotator Reaction Time pairwise stats (mean, sd, min, max, quantiles,...)


class ImageNetPsychophysics(object):
    """Dataloader for ImageNet with optional labels derived from the
    psychophysics oddball experiment that used ImageNet.
    More statistical information is available per class and sample to better
    inform the loss or for observation of relationships, if any between, the
    variables.

    Attributes
    ----------
    """

    # TODO load from csv, tsv, json, postgresql db

    # TODO save to csv, tsv, json, postgresql db

    # TODO get_item

    # Dataloader hsa everything needed for psychophysics:
    #   all Annotator RT info separate from instance level
    #   all Annotator confusion info separate from instance level
    #       class confusion matrix

# AssignmentId
# survey_index
# date_time
# ParentClusterId
# ChildClusterId
# worker_response
# ImposterFound
# ImageList
# ResponseTime

#class PairwiseReactionTime(object):
#    pass

class PsychophysicsDataLoader(object):

    def __init__(path):
        # TODO load from csv, tsv, json, postgresql db


        # FIRST, JUST LOAD THE RAW/PROCESSED DATA TO BE HANDLED! same format.


    # TODO save to csv, tsv, json, postgresql db

    # TODO get_item
    #   TODO given set of image samples from json format, obtain index from tsv/db

    # TODO Data management / visuals / utils
    #   TODO code to obtain human confusion matrix
    #       - for known exp: if correct then row == col, else row = other.
    #       - for unknown exp: if correct then row == col, else row = other.
    #           - loses pairwise info when correct...
    #   TODO code to obtain human reaction time "confusion matrix" or pair wise for
    #   classes
    #   TODO Code for annotator reliability?
    #       - Cannot rely on annotator amount to determine inter-reliability alone,
    #       needs Imagenet trusted labels w/ some degree of belief because we let
    #       the same annotator repeat the survey (hopefully a different survey, I
    #       believe that is the case), w/o tracking the annotator across surveys.
    #   TODO Code for specific instance confusion / RT?
    #   TODO Code for confusion / RT over time (ordinal by question)?
    #   TODO Code/label for support: number of samples per question?
    #       - support of images, classes, pairwise classes, etc. (only 25 per "annotator")
    #       - confirm support is the correct term and concept


def process_raw_csv(
    raw_csv_path,
    control_filepath,
    start_row=0,
    output_path=None,
    worker_id_col='AssignmentId',
):
    """Process the raw CSV version of the Amazon Turk annotators, but preserve
    some information about the annotator, such as their performance on control
    questions.

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

    [Args]
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
    else:
        raise ValueError('Expected filetype of npy or yaml')

    # Get rid of the "[" and "]" in the image list
    raw_data['ImageList'] = raw_data['ImageList'].apply(lambda x: x[1:-1])

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
        if nb_responses >= 27:
            raw_data = raw_data.drop(
                raw_data[raw_data[worker_id_col] == worker_id].index
            )

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


def annotator_control_question_score():
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
    # TODO load csv

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
    """
    raise NotImplementedError()

    # TODO load csv

    # TODO Per unique annotator, get confusion matrix when class is known /
    # unknown or host / imposter.
    # TODO Save the resulting confusion tensor w/ the ordered labels:
    ConfusionMatrices(targets, preds, labels, dim_vars).save(output_path)
    # Technically it is still a tensor, but not a complete confusion tensor.
