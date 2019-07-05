from src.core.Preparer import Preparer
from src.core.Learner import Learner

data_prepared = Preparer() \
    .show_insights() \
    .remove_redundant_cols() \
    .handle_nan() \
    .handle_name() \
    .handle_siblings() \
    .handle_categorical() \
    .build()

clf = Learner(data_prepared) \
    .selection() \
    .pick_best()
