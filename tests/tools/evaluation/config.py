# This file is part of culebra.
#
# Culebra is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Culebra is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# Culebra. If not, see <http://www.gnu.org/licenses/>.
#
# This work is supported by projects PGC2018-098813-B-C31 and
# PID2022-137461NB-C31, both funded by the Spanish "Ministerio de Ciencia,
# Innovación y Universidades" and by the European Regional Development Fund
# (ERDF).

"""Configuration file for the :class:`~culebra.tools.Evaluation` class."""

from collections import Counter

from sklearn.svm import SVC

from culebra.solution.feature_selection import (
    Species as FeatureSelectionSpecies,
    BitVector as FeatureSelectionIndividual
)
from culebra.solution.parameter_optimization import (
    Species as ClassifierOptimizationSpecies,
    Individual as ClassifierOptimizationIndividual
)
from culebra.fitness_function.feature_selection import (
    KappaIndex,
    NumFeats
)
from culebra.fitness_function.svc_optimization import C
from culebra.fitness_function.cooperative import FSSVCScorer
from culebra.trainer.ea import ElitistEA
from culebra.trainer.ea import ParallelCooperativeEA
from culebra.tools import Dataset


# Fitness function
def KappaNumFeatsC(training_data, test_data=None, cv_folds=None):
    """Fitness Function."""
    return FSSVCScorer(
        KappaIndex(
            training_data=training_data,
            test_data=test_data,
            classifier=SVC(kernel='rbf'),
            cv_folds=cv_folds
        ),
        NumFeats(),
        C()
    )


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Preprocess the dataset
dataset = dataset.drop_missing().scale().remove_outliers(random_seed=0)

# Split the dataset
(training_data, test_data) = dataset.split(test_prop=0.3, random_seed=0)

# Oversample the training data to make all the clases have the same number
# of samples
training_data = training_data.oversample(random_seed=0)


# Training fitness function
training_fitness_function = KappaNumFeatsC(training_data, cv_folds=5)

# Set the training fitness similarity threshold
training_fitness_function.obj_thresholds = 0.001

# Untie fitness function to select the best solution
samples_per_class = Counter(training_data.outputs)
max_folds = samples_per_class[
    min(samples_per_class, key=samples_per_class.get)
]
untie_best_fitness_function = KappaNumFeatsC(training_data, cv_folds=max_folds)

# Test fitness function
test_fitness_function = KappaNumFeatsC(training_data, test_data)

# Species to optimize a SVM-based classifier
classifierOptimizationSpecies = ClassifierOptimizationSpecies(
    lower_bounds=[0, 0],
    upper_bounds=[1000, 1000],
    names=["C", "gamma"]
)

# Species for the feature selection problem
featureSelectionSpecies1 = FeatureSelectionSpecies(
    num_feats=dataset.num_feats,
    max_feat=dataset.num_feats//2,
)
featureSelectionSpecies2 = FeatureSelectionSpecies(
    num_feats=dataset.num_feats,
    min_feat=dataset.num_feats//2 + 1,
)

# Parameters for the wrapper
params = {
    "solution_classes": [
        ClassifierOptimizationIndividual,
        FeatureSelectionIndividual,
        FeatureSelectionIndividual
    ],
    "species": [
        classifierOptimizationSpecies,
        featureSelectionSpecies1,
        featureSelectionSpecies2
    ],
    "fitness_function": training_fitness_function,
    "subtrainer_cls": ElitistEA,
    "representation_size": 2,
    "gene_ind_mutation_probs": (
        # At least one hyperparameter/feature will be mutated
        1.0/classifierOptimizationSpecies.num_params,
        2.0/dataset.num_feats,
        2.0/dataset.num_feats
    ),
    "max_num_iters": 100,
    "pop_sizes": dataset.num_feats//2,
    "checkpoint_enable": False,
}

# Create the trainer
trainer = ParallelCooperativeEA(**params)

# Define a base filename for the results
results_base_filename = "my_results"

# Require some custom parameters to appear in the results
hyperparameters = {"representation_size": 2, "max_num_iters": 100}
