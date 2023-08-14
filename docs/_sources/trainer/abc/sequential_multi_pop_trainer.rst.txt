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

:py:class:`culebra.trainer.abc.SequentialMultiPopTrainer` class
===============================================================

.. autoclass:: culebra.trainer.abc.SequentialMultiPopTrainer

Class attributes
----------------
.. autoattribute:: culebra.trainer.abc.SequentialMultiPopTrainer.stats_names
.. autoattribute:: culebra.trainer.abc.SequentialMultiPopTrainer.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.fitness_function
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.subpop_trainer_cls
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.max_num_iters
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.current_iter
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.custom_termination_func
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.num_subpops
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.representation_size
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.representation_freq
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.representation_topology_func
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.representation_topology_func_params
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.representation_selection_func
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.representation_selection_func_params
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.checkpoint_enable
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.checkpoint_freq
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.checkpoint_filename
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.verbose
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.random_seed
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.logbook
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.num_evals
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.runtime
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.index
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.container
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.representatives
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.subpop_trainer_params
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.subpop_trainer_checkpoint_filenames
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer.subpop_trainers


Private properties
------------------
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer._state
.. autoproperty:: culebra.trainer.abc.SequentialMultiPopTrainer._subpop_suffixes

Static methods
--------------
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer.receive_representatives
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer.send_representatives

Methods
-------
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer.reset
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer.evaluate
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer.best_solutions
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer.best_representatives
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer.train
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer.test

Private methods
---------------
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer._generate_subpop_trainers
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer._save_state
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer._load_state
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer._new_state
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer._init_state
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer._reset_state
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer._init_internals
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer._reset_internals
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer._init_search
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer._search
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer._finish_search
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer._start_iteration
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer._preprocess_iteration
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer._do_iteration
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer._postprocess_iteration
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer._finish_iteration
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer._do_iteration_stats
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer._default_termination_func
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer._termination_criterion
.. automethod:: culebra.trainer.abc.SequentialMultiPopTrainer._init_representatives
