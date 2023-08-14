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

:py:class:`culebra.trainer.abc.MultiPopTrainer` class
=====================================================

.. autoclass:: culebra.trainer.abc.MultiPopTrainer

Class attributes
----------------
.. autoattribute:: culebra.trainer.abc.MultiPopTrainer.stats_names
.. autoattribute:: culebra.trainer.abc.MultiPopTrainer.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.fitness_function
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.subpop_trainer_cls
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.max_num_iters
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.current_iter
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.custom_termination_func
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.num_subpops
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.representation_size
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.representation_freq
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.representation_topology_func
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.representation_topology_func_params
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.representation_selection_func
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.representation_selection_func_params
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.checkpoint_enable
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.checkpoint_freq
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.checkpoint_filename
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.verbose
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.random_seed
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.logbook
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.num_evals
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.runtime
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.index
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.container
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.representatives
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.subpop_trainer_params
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.subpop_trainer_checkpoint_filenames
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer.subpop_trainers


Private properties
------------------
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer._state
.. autoproperty:: culebra.trainer.abc.MultiPopTrainer._subpop_suffixes

Static methods
--------------
.. automethod:: culebra.trainer.abc.MultiPopTrainer.receive_representatives
.. automethod:: culebra.trainer.abc.MultiPopTrainer.send_representatives

Methods
-------
.. automethod:: culebra.trainer.abc.MultiPopTrainer.reset
.. automethod:: culebra.trainer.abc.MultiPopTrainer.evaluate
.. automethod:: culebra.trainer.abc.MultiPopTrainer.best_solutions
.. automethod:: culebra.trainer.abc.MultiPopTrainer.best_representatives
.. automethod:: culebra.trainer.abc.MultiPopTrainer.train
.. automethod:: culebra.trainer.abc.MultiPopTrainer.test

Private methods
---------------
.. automethod:: culebra.trainer.abc.MultiPopTrainer._generate_subpop_trainers
.. automethod:: culebra.trainer.abc.MultiPopTrainer._save_state
.. automethod:: culebra.trainer.abc.MultiPopTrainer._load_state
.. automethod:: culebra.trainer.abc.MultiPopTrainer._new_state
.. automethod:: culebra.trainer.abc.MultiPopTrainer._init_state
.. automethod:: culebra.trainer.abc.MultiPopTrainer._reset_state
.. automethod:: culebra.trainer.abc.MultiPopTrainer._init_internals
.. automethod:: culebra.trainer.abc.MultiPopTrainer._reset_internals
.. automethod:: culebra.trainer.abc.MultiPopTrainer._init_search
.. automethod:: culebra.trainer.abc.MultiPopTrainer._search
.. automethod:: culebra.trainer.abc.MultiPopTrainer._finish_search
.. automethod:: culebra.trainer.abc.MultiPopTrainer._start_iteration
.. automethod:: culebra.trainer.abc.MultiPopTrainer._preprocess_iteration
.. automethod:: culebra.trainer.abc.MultiPopTrainer._do_iteration
.. automethod:: culebra.trainer.abc.MultiPopTrainer._postprocess_iteration
.. automethod:: culebra.trainer.abc.MultiPopTrainer._finish_iteration
.. automethod:: culebra.trainer.abc.MultiPopTrainer._do_iteration_stats
.. automethod:: culebra.trainer.abc.MultiPopTrainer._default_termination_func
.. automethod:: culebra.trainer.abc.MultiPopTrainer._termination_criterion
.. automethod:: culebra.trainer.abc.MultiPopTrainer._init_representatives
