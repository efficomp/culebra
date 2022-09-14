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

:py:mod:`base` module
=====================

.. automodule:: base


Attributes
----------
.. attribute:: DEFAULT_SEP
    :annotation: = '\\s+'

    Default column separator used within dataset files.

.. attribute:: DEFAULT_STATS_NAMES
    :annotation: = ('Iter', 'NEvals')

    Default statistics calculated for each iteration of the wrapper.

.. attribute:: DEFAULT_OBJECTIVE_STATS
    :annotation: = {'Avg': <function mean>, 'Max': <function amax>, 'Min': <function amin>, 'Std': <function std>}

    Default statistics calculated for each objective within a wrapper.

.. attribute:: DEFAULT_CHECKPOINT_ENABLE
    :annotation: = True

    Default checkpoinitng enablement for wrappers.

.. attribute:: DEFAULT_CHECKPOINT_FREQ
    :annotation: = 10

    Default checkpointing frequency for wrappers.

.. attribute:: DEFAULT_CHECKPOINT_FILENAME
    :annotation: = 'checkpoint.gz'

    Default checkpointing file name for wrappers.

.. attribute:: DEFAULT_VERBOSITY
    :annotation: = True

    Default verbosity for wrappers.

.. attribute:: DEFAULT_INDEX
    :annotation: = 0

    Default wrapper index. Only used within distributed wrappers.

.. attribute:: DEFAULT_CLASSIFIER
    :annotation: = <class 'sklearn.naive_bayes.GaussianNB'>

    Default classifier for fitness functions.

..
    .. autodata:: DEFAULT_SEP
    .. autodata:: DEFAULT_STATS_NAMES
    .. autodata:: DEFAULT_OBJECTIVE_STATS
    .. autodata:: DEFAULT_CHECKPOINT_ENABLE
    .. autodata:: DEFAULT_CHECKPOINT_FREQ
    .. autodata:: DEFAULT_CHECKPOINT_FILENAME
    .. autodata:: DEFAULT_VERBOSITY
    .. autodata:: DEFAULT_INDEX
    .. autodata:: DEFAULT_CLASSIFIER

.. toctree::
    :hidden:

    base <base/base>
    dataset <base/dataset>
    individual <base/individual>
    species <base/species>
    fitness_function <base/fitness_function>
    fitness <base/fitness>
    wrapper <base/wrapper>
    checker <base/checker>
