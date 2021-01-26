import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('preprocessed_data.csv')
features = tpot_data.drop('Emotion', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Emotion'], random_state=42)

# Average CV score on the training set was: 0.7454545454545455
exported_pipeline = make_pipeline(
    VarianceThreshold(threshold=0.0001),
    RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.05, n_estimators=100), step=0.15000000000000002),
    StackingEstimator(estimator=LinearSVC(C=25.0, dual=True, loss="squared_hinge", penalty="l2", tol=1e-05)),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=4, max_features=0.6000000000000001, min_samples_leaf=1, min_samples_split=8, n_estimators=100, subsample=0.9500000000000001)
)

# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

# Mean Accuracy
print('Mean accuracy: %r' % np.mean(results == testing_target.values))