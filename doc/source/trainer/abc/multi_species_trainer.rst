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

:py:class:`culebra.trainer.abc.MultiSpeciesTrainer` class
=========================================================

.. autoclass:: culebra.trainer.abc.MultiSpeciesTrainer

Class attributes
----------------
.. autoattribute:: culebra.trainer.abc.MultiSpeciesTrainer.stats_names
.. autoattribute:: culebra.trainer.abc.MultiSpeciesTrainer.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.abc.MultiSpeciesTrainer.solution_classes
.. autoproperty:: culebra.trainer.abc.MultiSpeciesTrainer.species
.. autoproperty:: culebra.trainer.abc.MultiSpeciesTrainer.fitness_function
.. autoproperty:: culebra.trainer.abc.MultiSpeciesTrainer.max_num_iters
.. autoproperty:: culebra.trainer.abc.MultiSpeciesTrainer.current_iter
.. autoproperty:: culebra.trainer.abc.MultiSpeciesTrainer.custom_termination_func
.. autoproperty:: culebra.trainer.abc.MultiSpeciesTrainer.checkpoint_enable
.. autoproperty:: culebra.trainer.abc.MultiSpeciesTrainer.checkpoint_freq
.. autoproperty:: culebra.trainer.abc.MultiSpeciesTrainer.checkpoint_filename
.. autoproperty:: culebra.trainer.abc.MultiSpeciesTrainer.verbose
.. autoproperty:: culebra.trainer.abc.MultiSpeciesTrainer.random_seed
.. autoproperty:: culebra.trainer.abc.MultiSpeciesTrainer.logbook
.. autoproperty:: culebra.trainer.abc.MultiSpeciesTrainer.num_evals
.. autoproperty:: culebra.trainer.abc.MultiSpeciesTrainer.runtime
.. autoproperty:: culebra.trainer.abc.MultiSpeciesTrainer.index
.. autoproperty:: culebra.trainer.abc.MultiSpeciesTrainer.container
.. autoproperty:: culebra.trainer.abc.MultiSpeciesTrainer.representatives

Private properties
------------------
.. autoproperty:: culebra.trainer.abc.MultiSpeciesTrainer._state

Methods
-------
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer.reset
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer.evaluate
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer.best_solutions
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer.best_representatives
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer.train
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer.test

Private methods
---------------
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer._save_state
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer._load_state
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer._new_state
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer._init_state
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer._reset_state
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer._init_internals
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer._reset_internals
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer._init_search
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer._search
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer._finish_search
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer._start_iteration
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer._preprocess_iteration
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer._do_iteration
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer._postprocess_iteration
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer._finish_iteration
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer._do_iteration_stats
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer._default_termination_func
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer._termination_criterion
.. automethod:: culebra.trainer.abc.MultiSpeciesTrainer._init_representatives
