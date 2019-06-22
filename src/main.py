from src.Preparer import Preparer
from src.Learner import Learner

data_prepared = Preparer() \
    .remove_redundant_cols() \
    .handle_nan() \
    .handle_categorical() \
    .build()

Learner(data_prepared).check_svm()
