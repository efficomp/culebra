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

:py:class:`culebra.trainer.ea.SimpleEA` class
=============================================

.. autoclass:: culebra.trainer.ea.SimpleEA

Class attributes
----------------
.. autoattribute:: culebra.trainer.ea.SimpleEA.stats_names
.. autoattribute:: culebra.trainer.ea.SimpleEA.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.ea.SimpleEA.solution_cls
.. autoproperty:: culebra.trainer.ea.SimpleEA.species
.. autoproperty:: culebra.trainer.ea.SimpleEA.fitness_function
.. autoproperty:: culebra.trainer.ea.SimpleEA.max_num_iters
.. autoproperty:: culebra.trainer.ea.SimpleEA.current_iter
.. autoproperty:: culebra.trainer.ea.SimpleEA.custom_termination_func
.. autoproperty:: culebra.trainer.ea.SimpleEA.pop_size
.. autoproperty:: culebra.trainer.ea.SimpleEA.crossover_func
.. autoproperty:: culebra.trainer.ea.SimpleEA.mutation_func
.. autoproperty:: culebra.trainer.ea.SimpleEA.selection_func
.. autoproperty:: culebra.trainer.ea.SimpleEA.crossover_prob
.. autoproperty:: culebra.trainer.ea.SimpleEA.mutation_prob
.. autoproperty:: culebra.trainer.ea.SimpleEA.gene_ind_mutation_prob
.. autoproperty:: culebra.trainer.ea.SimpleEA.selection_func_params
.. autoproperty:: culebra.trainer.ea.SimpleEA.checkpoint_enable
.. autoproperty:: culebra.trainer.ea.SimpleEA.checkpoint_freq
.. autoproperty:: culebra.trainer.ea.SimpleEA.checkpoint_filename
.. autoproperty:: culebra.trainer.ea.SimpleEA.verbose
.. autoproperty:: culebra.trainer.ea.SimpleEA.random_seed
.. autoproperty:: culebra.trainer.ea.SimpleEA.logbook
.. autoproperty:: culebra.trainer.ea.SimpleEA.num_evals
.. autoproperty:: culebra.trainer.ea.SimpleEA.runtime
.. autoproperty:: culebra.trainer.ea.SimpleEA.index
.. autoproperty:: culebra.trainer.ea.SimpleEA.container
.. autoproperty:: culebra.trainer.ea.SimpleEA.representatives
.. autoproperty:: culebra.trainer.ea.SimpleEA.pop

Methods
-------
.. automethod:: culebra.trainer.ea.SimpleEA.reset
.. automethod:: culebra.trainer.ea.SimpleEA.evaluate
.. automethod:: culebra.trainer.ea.SimpleEA.best_solutions
.. automethod:: culebra.trainer.ea.SimpleEA.best_representatives
.. automethod:: culebra.trainer.ea.SimpleEA.train
.. automethod:: culebra.trainer.ea.SimpleEA.test

Private methods
---------------
.. automethod:: culebra.trainer.ea.SimpleEA._get_state
.. automethod:: culebra.trainer.ea.SimpleEA._set_state
.. automethod:: culebra.trainer.ea.SimpleEA._save_state
.. automethod:: culebra.trainer.ea.SimpleEA._load_state
.. automethod:: culebra.trainer.ea.SimpleEA._new_state
.. automethod:: culebra.trainer.ea.SimpleEA._init_state
.. automethod:: culebra.trainer.ea.SimpleEA._reset_state
.. automethod:: culebra.trainer.ea.SimpleEA._init_internals
.. automethod:: culebra.trainer.ea.SimpleEA._reset_internals
.. automethod:: culebra.trainer.ea.SimpleEA._init_search
.. automethod:: culebra.trainer.ea.SimpleEA._search
.. automethod:: culebra.trainer.ea.SimpleEA._finish_search
.. automethod:: culebra.trainer.ea.SimpleEA._start_iteration
.. automethod:: culebra.trainer.ea.SimpleEA._preprocess_iteration
.. automethod:: culebra.trainer.ea.SimpleEA._do_iteration
.. automethod:: culebra.trainer.ea.SimpleEA._postprocess_iteration
.. automethod:: culebra.trainer.ea.SimpleEA._finish_iteration
.. automethod:: culebra.trainer.ea.SimpleEA._do_iteration_stats
.. automethod:: culebra.trainer.ea.SimpleEA._default_termination_func
.. automethod:: culebra.trainer.ea.SimpleEA._termination_criterion
.. automethod:: culebra.trainer.ea.SimpleEA._init_representatives
.. automethod:: culebra.trainer.ea.SimpleEA._generate_initial_pop
.. automethod:: culebra.trainer.ea.SimpleEA._evaluate_pop
