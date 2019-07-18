from src.core.Preparer import Preparer
from src.core.Learner import Learner
from src.constants import src_models

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
    .save_best(src_models+"titanic_model.joblib") \
    .pick_best()
