from src.Preparer import Preparer
from src.Learner import Learner

data_prepared = Preparer() \
    .show_insights() \
    .remove_redundant_cols() \
    .handle_nan() \
    .handle_categorical() \
    .build()

clf = Learner(data_prepared) \
    .selection() \
    .pick_best()
