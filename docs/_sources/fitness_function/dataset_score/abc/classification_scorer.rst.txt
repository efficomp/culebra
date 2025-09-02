..
   This file is part of

   Culebra is free software: you can redistribute it and/or modify it under the
   terms of the GNU General Public License as published by the Free Software
   Foundation, either version 3 of the License, or (at your option) any later
   version.

   Culebra is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
   details.

   You should have received a copy of the GNU General Public License along with
   Culebra. If not, see <http://www.gnu.org/licenses/>.

   This work is supported by projects PGC2018-098813-B-C31 and
   PID2022-137461NB-C31, both funded by the Spanish "Ministerio de Ciencia,
   Innovación y Universidades" and by the European Regional Development Fund
   (ERDF).

:py:class:`culebra.fitness_function.dataset_score.abc.ClassificationScorer` class
=================================================================================

.. autoclass:: culebra.fitness_function.dataset_score.abc.ClassificationScorer

Class methods
-------------
.. automethod:: culebra.fitness_function.dataset_score.abc.ClassificationScorer.load

Properties
----------
.. autoproperty:: culebra.fitness_function.dataset_score.abc.ClassificationScorer.num_obj
.. autoproperty:: culebra.fitness_function.dataset_score.abc.ClassificationScorer.obj_weights
.. autoproperty:: culebra.fitness_function.dataset_score.abc.ClassificationScorer.obj_names
.. autoproperty:: culebra.fitness_function.dataset_score.abc.ClassificationScorer.obj_thresholds
.. autoproperty:: culebra.fitness_function.dataset_score.abc.ClassificationScorer.fitness_cls
.. autoproperty:: culebra.fitness_function.dataset_score.abc.ClassificationScorer.is_noisy
.. autoproperty:: culebra.fitness_function.dataset_score.abc.ClassificationScorer.index
.. autoproperty:: culebra.fitness_function.dataset_score.abc.ClassificationScorer.training_data
.. autoproperty:: culebra.fitness_function.dataset_score.abc.ClassificationScorer.test_data
.. autoproperty:: culebra.fitness_function.dataset_score.abc.ClassificationScorer.test_prop
.. autoproperty:: culebra.fitness_function.dataset_score.abc.ClassificationScorer.classifier

Private properties
------------------
.. autoproperty:: culebra.fitness_function.dataset_score.abc.ClassificationScorer._worst_score

Methods
-------
.. automethod:: culebra.fitness_function.dataset_score.abc.ClassificationScorer.dump
.. automethod:: culebra.fitness_function.dataset_score.abc.ClassificationScorer.evaluate
.. automethod:: culebra.fitness_function.dataset_score.abc.ClassificationScorer.is_evaluable

Private methods
---------------
.. automethod:: culebra.fitness_function.dataset_score.abc.ClassificationScorer._score
.. automethod:: culebra.fitness_function.dataset_score.abc.ClassificationScorer._final_training_test_data
.. automethod:: culebra.fitness_function.dataset_score.abc.ClassificationScorer._evaluate_train_test
.. automethod:: culebra.fitness_function.dataset_score.abc.ClassificationScorer._evaluate_mccv
.. automethod:: culebra.fitness_function.dataset_score.abc.ClassificationScorer._evaluate_kfcv
