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

:py:class:`base.Wrapper` class
==============================

.. autoclass:: base.Wrapper

Class attributes
----------------
.. autoattribute:: base.Wrapper.stats_names
.. autoattribute:: base.Wrapper.objective_stats

Properties
----------
.. autoproperty:: base.Wrapper.fitness_function
.. autoproperty:: base.Wrapper.checkpoint_enable
.. autoproperty:: base.Wrapper.checkpoint_freq
.. autoproperty:: base.Wrapper.checkpoint_filename
.. autoproperty:: base.Wrapper.verbose
.. autoproperty:: base.Wrapper.random_seed
.. autoproperty:: base.Wrapper.logbook
.. autoproperty:: base.Wrapper.num_evals
.. autoproperty:: base.Wrapper.runtime
.. autoproperty:: base.Wrapper.index
.. autoproperty:: base.Wrapper.container
.. autoproperty:: base.Wrapper.representatives

Private Properties
------------------
.. autoproperty:: base.Wrapper._state

Methods
-------
.. automethod:: base.Wrapper.reset
.. automethod:: base.Wrapper.evaluate
.. automethod:: base.Wrapper.best_solutions
.. automethod:: base.Wrapper.best_representatives
.. automethod:: base.Wrapper.train
.. automethod:: base.Wrapper.test

Private methods
---------------
.. automethod:: base.Wrapper._save_state
.. automethod:: base.Wrapper._load_state
.. automethod:: base.Wrapper._new_state
.. automethod:: base.Wrapper._reset_state
.. automethod:: base.Wrapper._init_state
.. automethod:: base.Wrapper._init_internals
.. automethod:: base.Wrapper._reset_internals
.. automethod:: base.Wrapper._init_search
.. automethod:: base.Wrapper._search
.. automethod:: base.Wrapper._finish_search
.. automethod:: base.Wrapper._init_representatives
