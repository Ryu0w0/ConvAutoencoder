from enum import auto, Enum


class TrainType(Enum):
    CV_TRAIN = auto()
    CV_VALID = auto()


# cross-validation training mode
cv_train = "train"
cv_valid = "valid"
