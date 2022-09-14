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

:py:class:`wrapper.multi_pop.ParallelMultiPop` class
====================================================

.. autoclass:: wrapper.multi_pop.ParallelMultiPop

Class attributes
----------------
.. autoattribute:: wrapper.multi_pop.ParallelMultiPop.stats_names
.. autoattribute:: wrapper.multi_pop.ParallelMultiPop.objective_stats

Properties
----------
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.fitness_function
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.subpop_wrapper_cls
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.subpop_wrapper_params
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.num_gens
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.num_subpops
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.representation_size
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.representation_freq
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.representation_topology_func
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.representation_topology_func_params
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.representation_selection_func
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.representation_selection_func_params
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.checkpoint_enable
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.checkpoint_freq
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.checkpoint_filename
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.verbose
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.random_seed
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.subpop_wrappers
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.subpop_wrapper_checkpoint_filenames
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.logbook
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.num_evals
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.runtime
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.index
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.container
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop.representatives

Private Properties
------------------
.. autoproperty:: wrapper.multi_pop.ParallelMultiPop._state

Static methods
--------------
.. automethod:: wrapper.multi_pop.ParallelMultiPop.receive_representatives
.. automethod:: wrapper.multi_pop.ParallelMultiPop.send_representatives


Methods
-------
.. automethod:: wrapper.multi_pop.ParallelMultiPop.reset
.. automethod:: wrapper.multi_pop.ParallelMultiPop.evaluate
.. automethod:: wrapper.multi_pop.ParallelMultiPop.best_solutions
.. automethod:: wrapper.multi_pop.ParallelMultiPop.best_representatives
.. automethod:: wrapper.multi_pop.ParallelMultiPop.train
.. automethod:: wrapper.multi_pop.ParallelMultiPop.test


Private methods
---------------
.. automethod:: wrapper.multi_pop.ParallelMultiPop._generate_subpop_wrappers
.. automethod:: wrapper.multi_pop.ParallelMultiPop._save_state
.. automethod:: wrapper.multi_pop.ParallelMultiPop._load_state
.. automethod:: wrapper.multi_pop.ParallelMultiPop._new_state
.. automethod:: wrapper.multi_pop.ParallelMultiPop._reset_state
.. automethod:: wrapper.multi_pop.ParallelMultiPop._init_state
.. automethod:: wrapper.multi_pop.ParallelMultiPop._init_internals
.. automethod:: wrapper.multi_pop.ParallelMultiPop._reset_internals
.. automethod:: wrapper.multi_pop.ParallelMultiPop._init_search
.. automethod:: wrapper.multi_pop.ParallelMultiPop._start_generation
.. automethod:: wrapper.multi_pop.ParallelMultiPop._preprocess_generation
.. automethod:: wrapper.multi_pop.ParallelMultiPop._do_generation
.. automethod:: wrapper.multi_pop.ParallelMultiPop._postprocess_generation
.. automethod:: wrapper.multi_pop.ParallelMultiPop._finish_generation
.. automethod:: wrapper.multi_pop.ParallelMultiPop._search
.. automethod:: wrapper.multi_pop.ParallelMultiPop._finish_search
.. automethod:: wrapper.multi_pop.ParallelMultiPop._init_representatives
