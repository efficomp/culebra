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

:py:class:`culebra.trainer.abc.ParallelMultiPopTrainer` class
=============================================================

.. autoclass:: culebra.trainer.abc.ParallelMultiPopTrainer

Class attributes
----------------
.. autoattribute:: culebra.trainer.abc.ParallelMultiPopTrainer.stats_names
.. autoattribute:: culebra.trainer.abc.ParallelMultiPopTrainer.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.fitness_function
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.subpop_trainer_cls
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.max_num_iters
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.current_iter
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.custom_termination_func
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.num_subpops
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.representation_size
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.representation_freq
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.representation_topology_func
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.representation_topology_func_params
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.representation_selection_func
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.representation_selection_func_params
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.checkpoint_enable
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.checkpoint_freq
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.checkpoint_filename
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.verbose
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.random_seed
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.logbook
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.num_evals
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.runtime
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.index
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.container
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.representatives
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.subpop_trainer_params
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.subpop_trainer_checkpoint_filenames
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer.subpop_trainers


Private properties
------------------
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer._state
.. autoproperty:: culebra.trainer.abc.ParallelMultiPopTrainer._subpop_suffixes

Static methods
--------------
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer.receive_representatives
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer.send_representatives

Methods
-------
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer.reset
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer.evaluate
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer.best_solutions
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer.best_representatives
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer.train
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer.test

Private methods
---------------
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer._generate_subpop_trainers
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer._save_state
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer._load_state
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer._new_state
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer._init_state
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer._reset_state
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer._init_internals
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer._reset_internals
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer._init_search
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer._search
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer._finish_search
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer._start_iteration
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer._preprocess_iteration
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer._do_iteration
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer._postprocess_iteration
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer._finish_iteration
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer._do_iteration_stats
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer._default_termination_func
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer._termination_criterion
.. automethod:: culebra.trainer.abc.ParallelMultiPopTrainer._init_representatives
