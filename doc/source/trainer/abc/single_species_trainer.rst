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

:py:class:`culebra.trainer.abc.SingleSpeciesTrainer` class
==========================================================

.. autoclass:: culebra.trainer.abc.SingleSpeciesTrainer

Class attributes
----------------
.. autoattribute:: culebra.trainer.abc.SingleSpeciesTrainer.stats_names
.. autoattribute:: culebra.trainer.abc.SingleSpeciesTrainer.objective_stats

Class methods
-------------
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer.load

Properties
----------
.. autoproperty:: culebra.trainer.abc.SingleSpeciesTrainer.solution_cls
.. autoproperty:: culebra.trainer.abc.SingleSpeciesTrainer.species
.. autoproperty:: culebra.trainer.abc.SingleSpeciesTrainer.fitness_function
.. autoproperty:: culebra.trainer.abc.SingleSpeciesTrainer.max_num_iters
.. autoproperty:: culebra.trainer.abc.SingleSpeciesTrainer.current_iter
.. autoproperty:: culebra.trainer.abc.SingleSpeciesTrainer.custom_termination_func
.. autoproperty:: culebra.trainer.abc.SingleSpeciesTrainer.checkpoint_enable
.. autoproperty:: culebra.trainer.abc.SingleSpeciesTrainer.checkpoint_freq
.. autoproperty:: culebra.trainer.abc.SingleSpeciesTrainer.checkpoint_filename
.. autoproperty:: culebra.trainer.abc.SingleSpeciesTrainer.verbose
.. autoproperty:: culebra.trainer.abc.SingleSpeciesTrainer.random_seed
.. autoproperty:: culebra.trainer.abc.SingleSpeciesTrainer.logbook
.. autoproperty:: culebra.trainer.abc.SingleSpeciesTrainer.num_evals
.. autoproperty:: culebra.trainer.abc.SingleSpeciesTrainer.runtime
.. autoproperty:: culebra.trainer.abc.SingleSpeciesTrainer.index
.. autoproperty:: culebra.trainer.abc.SingleSpeciesTrainer.container
.. autoproperty:: culebra.trainer.abc.SingleSpeciesTrainer.representatives

Methods
-------
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer.dump
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer.reset
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer.evaluate
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer.best_solutions
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer.best_representatives
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer.train
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer.test

Private methods
---------------
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._get_state
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._set_state
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._save_state
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._load_state
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._new_state
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._init_state
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._reset_state
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._init_internals
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._reset_internals
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._init_search
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._search
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._finish_search
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._start_iteration
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._preprocess_iteration
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._do_iteration
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._postprocess_iteration
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._finish_iteration
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._do_iteration_stats
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._default_termination_func
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._termination_criterion
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._init_representatives
.. automethod:: culebra.trainer.abc.SingleSpeciesTrainer._set_cooperative_fitness
