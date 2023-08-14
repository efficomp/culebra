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
    If not, see <http://www.gnu.org/licenses/>.

   This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
   de Ciencia, Innovación y Universidades"), and by the European Regional
   Development Fund (ERDF).

:py:class:`culebra.trainer.abc.SinglePopTrainer` class
======================================================

.. autoclass:: culebra.trainer.abc.SinglePopTrainer

Class attributes
----------------
.. autoattribute:: culebra.trainer.abc.SinglePopTrainer.stats_names
.. autoattribute:: culebra.trainer.abc.SinglePopTrainer.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.abc.SinglePopTrainer.solution_cls
.. autoproperty:: culebra.trainer.abc.SinglePopTrainer.species
.. autoproperty:: culebra.trainer.abc.SinglePopTrainer.fitness_function
.. autoproperty:: culebra.trainer.abc.SinglePopTrainer.max_num_iters
.. autoproperty:: culebra.trainer.abc.SinglePopTrainer.current_iter
.. autoproperty:: culebra.trainer.abc.SinglePopTrainer.custom_termination_func
.. autoproperty:: culebra.trainer.abc.SinglePopTrainer.pop_size
.. autoproperty:: culebra.trainer.abc.SinglePopTrainer.checkpoint_enable
.. autoproperty:: culebra.trainer.abc.SinglePopTrainer.checkpoint_freq
.. autoproperty:: culebra.trainer.abc.SinglePopTrainer.checkpoint_filename
.. autoproperty:: culebra.trainer.abc.SinglePopTrainer.verbose
.. autoproperty:: culebra.trainer.abc.SinglePopTrainer.random_seed
.. autoproperty:: culebra.trainer.abc.SinglePopTrainer.logbook
.. autoproperty:: culebra.trainer.abc.SinglePopTrainer.num_evals
.. autoproperty:: culebra.trainer.abc.SinglePopTrainer.runtime
.. autoproperty:: culebra.trainer.abc.SinglePopTrainer.index
.. autoproperty:: culebra.trainer.abc.SinglePopTrainer.container
.. autoproperty:: culebra.trainer.abc.SinglePopTrainer.representatives
.. autoproperty:: culebra.trainer.abc.SinglePopTrainer.pop

Private properties
------------------
.. autoproperty:: culebra.trainer.abc.SinglePopTrainer._state

Methods
-------
.. automethod:: culebra.trainer.abc.SinglePopTrainer.reset
.. automethod:: culebra.trainer.abc.SinglePopTrainer.evaluate
.. automethod:: culebra.trainer.abc.SinglePopTrainer.best_solutions
.. automethod:: culebra.trainer.abc.SinglePopTrainer.best_representatives
.. automethod:: culebra.trainer.abc.SinglePopTrainer.train
.. automethod:: culebra.trainer.abc.SinglePopTrainer.test

Private methods
---------------
.. automethod:: culebra.trainer.abc.SinglePopTrainer._save_state
.. automethod:: culebra.trainer.abc.SinglePopTrainer._load_state
.. automethod:: culebra.trainer.abc.SinglePopTrainer._new_state
.. automethod:: culebra.trainer.abc.SinglePopTrainer._init_state
.. automethod:: culebra.trainer.abc.SinglePopTrainer._reset_state
.. automethod:: culebra.trainer.abc.SinglePopTrainer._init_internals
.. automethod:: culebra.trainer.abc.SinglePopTrainer._reset_internals
.. automethod:: culebra.trainer.abc.SinglePopTrainer._init_search
.. automethod:: culebra.trainer.abc.SinglePopTrainer._search
.. automethod:: culebra.trainer.abc.SinglePopTrainer._finish_search
.. automethod:: culebra.trainer.abc.SinglePopTrainer._start_iteration
.. automethod:: culebra.trainer.abc.SinglePopTrainer._preprocess_iteration
.. automethod:: culebra.trainer.abc.SinglePopTrainer._do_iteration
.. automethod:: culebra.trainer.abc.SinglePopTrainer._postprocess_iteration
.. automethod:: culebra.trainer.abc.SinglePopTrainer._finish_iteration
.. automethod:: culebra.trainer.abc.SinglePopTrainer._do_iteration_stats
.. automethod:: culebra.trainer.abc.SinglePopTrainer._default_termination_func
.. automethod:: culebra.trainer.abc.SinglePopTrainer._termination_criterion
.. automethod:: culebra.trainer.abc.SinglePopTrainer._init_representatives
