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
