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

:py:class:`culebra.trainer.ea.abc.HomogeneousEA` class
======================================================

.. autoclass:: culebra.trainer.ea.abc.HomogeneousEA

Class attributes
----------------
.. autoattribute:: culebra.trainer.ea.abc.HomogeneousEA.stats_names
.. autoattribute:: culebra.trainer.ea.abc.HomogeneousEA.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.solution_cls
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.species
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.fitness_function
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.max_num_iters
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.current_iter
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.custom_termination_func
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.pop_size
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.crossover_func
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.mutation_func
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.selection_func
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.crossover_prob
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.mutation_prob
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.gene_ind_mutation_prob
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.selection_func_params
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.checkpoint_enable
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.checkpoint_freq
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.checkpoint_filename
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.verbose
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.random_seed
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.logbook
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.num_evals
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.runtime
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.index
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.container
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA.representatives

Private properties
------------------
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousEA._state

Methods
-------
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA.reset
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA.evaluate
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA.best_solutions
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA.best_representatives
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA.train
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA.test

Private methods
---------------
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA._save_state
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA._load_state
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA._new_state
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA._init_state
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA._reset_state
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA._init_internals
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA._reset_internals
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA._init_search
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA._search
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA._finish_search
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA._start_iteration
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA._preprocess_iteration
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA._do_iteration
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA._postprocess_iteration
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA._finish_iteration
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA._do_iteration_stats
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA._default_termination_func
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA._termination_criterion
.. automethod:: culebra.trainer.ea.abc.HomogeneousEA._init_representatives
