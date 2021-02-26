"""Dataloader for psychophysics oddball experiment using ImageNet."""
import torch


class PsychophysicsDataLoader(object):
    """Dataloader for psychophysics oddball experiment using ImageNet.
    More statistical information is available per class and sample to better
    inform the loss.
    """

    # TODO load

# AssignmentId
# survey_index
# date_time
# ParentClusterId
# ChildClusterId
# worker_response
# ImposterFound
# ImageList
# ResponseTime

# TODO Data management / visuals / utils
#   TODO given set of image samples from json format, obtain index from tsv/db
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
