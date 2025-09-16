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

:py:class:`culebra.abc.Trainer` class
=====================================

.. autoclass:: culebra.abc.Trainer

Class attributes
----------------
.. autoattribute:: culebra.abc.Trainer.stats_names
.. autoattribute:: culebra.abc.Trainer.objective_stats

Class methods
-------------
.. automethod:: culebra.abc.Trainer.load

Properties
----------
.. autoproperty:: culebra.abc.Trainer.fitness_function
.. autoproperty:: culebra.abc.Trainer.max_num_iters
.. autoproperty:: culebra.abc.Trainer.current_iter
.. autoproperty:: culebra.abc.Trainer.custom_termination_func
.. autoproperty:: culebra.abc.Trainer.checkpoint_enable
.. autoproperty:: culebra.abc.Trainer.checkpoint_freq
.. autoproperty:: culebra.abc.Trainer.checkpoint_filename
.. autoproperty:: culebra.abc.Trainer.verbose
.. autoproperty:: culebra.abc.Trainer.random_seed
.. autoproperty:: culebra.abc.Trainer.logbook
.. autoproperty:: culebra.abc.Trainer.num_evals
.. autoproperty:: culebra.abc.Trainer.runtime
.. autoproperty:: culebra.abc.Trainer.index
.. autoproperty:: culebra.abc.Trainer.container
.. autoproperty:: culebra.abc.Trainer.representatives


Methods
-------
.. automethod:: culebra.abc.Trainer.dump
.. automethod:: culebra.abc.Trainer.reset
.. automethod:: culebra.abc.Trainer.evaluate
.. automethod:: culebra.abc.Trainer.best_solutions
.. automethod:: culebra.abc.Trainer.best_representatives
.. automethod:: culebra.abc.Trainer.train
.. automethod:: culebra.abc.Trainer.test

Private methods
---------------
.. automethod:: culebra.abc.Trainer._get_state
.. automethod:: culebra.abc.Trainer._set_state
.. automethod:: culebra.abc.Trainer._save_state
.. automethod:: culebra.abc.Trainer._load_state
.. automethod:: culebra.abc.Trainer._new_state
.. automethod:: culebra.abc.Trainer._init_state
.. automethod:: culebra.abc.Trainer._reset_state
.. automethod:: culebra.abc.Trainer._init_internals
.. automethod:: culebra.abc.Trainer._reset_internals
.. automethod:: culebra.abc.Trainer._init_search
.. automethod:: culebra.abc.Trainer._search
.. automethod:: culebra.abc.Trainer._finish_search
.. automethod:: culebra.abc.Trainer._start_iteration
.. automethod:: culebra.abc.Trainer._preprocess_iteration
.. automethod:: culebra.abc.Trainer._do_iteration
.. automethod:: culebra.abc.Trainer._postprocess_iteration
.. automethod:: culebra.abc.Trainer._finish_iteration
.. automethod:: culebra.abc.Trainer._do_iteration_stats
.. automethod:: culebra.abc.Trainer._default_termination_func
.. automethod:: culebra.abc.Trainer._termination_criterion
.. automethod:: culebra.abc.Trainer._init_representatives
.. automethod:: culebra.abc.Trainer._set_cooperative_fitness
