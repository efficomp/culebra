# Culebra: Metaheuristic Optimization Framework

[![License: GPL v3] (https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+] (https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Powered by DEAP] (https://img.shields.io/badge/powered%20by-DEAP-orange.svg)](https://github.com/DEAP/deap)


**Culebra** is a versatile and extensible Python framework for solving complex
optimization problems using metaheuristics. While it originated as a tool for
Feature Selection, it has evolved into a general-purpose engine supporting
multiple paradigms, including Evolutionary Algorithms (EA), Ant Colony
Optimization (ACO), and Multi-Species Coevolution.

---

## 🌟 Key Capabilities

Culebra is no longer limited to feature selection. Its modular architecture
allows you to tackle:

* **Distributed & Parallel Execution:** Native support for distributed
  versions, sequential or parallel, of any metaheuristic. The framework
  abstracts the execution layer, allowing seamless scaling.

* **Combinatorial Optimization:** Native support for problems like the
  *Traveling Salesman Problem (TSP)*.

* **Diverse Metaheuristics:**

  * **EA (Evolutionary Algorithms):** Standard and coevolutionary approaches.

  * **ACO (Ant Colony Optimization):** Specialized solvers for pathfinding and
    permutation problems.

* **Multi-Species Coevolution:** Advanced cooperative models where different
  populations evolve simultaneously to solve a global task.

* **Feature Selection:** High-performance wrappers for Scikit-learn,
  including specialized fitness functions for imbalanced data.

---

## 🛠️ Requirements

Culebra requires Python 3.8+. It also depends on the following Python packages:

* [DEAP] (https://pypi.org/project/deap/)
* [NumPy] (https://pypi.org/project/numpy/)
* [Pandas] (https://pypi.org/project/pandas/)
* [Scikit-learn] (https://pypi.org/project/scikit-learn/)
* [Sphinx] (https://pypi.org/project/Sphinx/)
* [Tabulate] (https://pypi.org/project/tabulate/)
* [OpenPyXL] (https://pypi.org/project/openpyxl/)
* [scikit-posthocs] (https://pypi.org/project/scikit-posthocs/)
* [ucimlrepo] (https://pypi.org/project/ucimlrepo/)
* [imbalanced-learn] (https://pypi.org/project/imbalanced-learn/)
* [dill] (https://pypi.org/project/dill/)
* [multiprocess] (https://pypi.org/project/multiprocess/)

---

## 📦 Installation
Culebra provides a `setup.sh` script to automate the environment configuration
and install all dependencies.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/efficomp/culebra.git](https://github.com/efficomp/culebra.git)
    cd culebra
    ```

2.  **Run the setup script:**
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```
---

## 📖 Documentation

Culebra is fully documented in its [github-pages] (https://efficomp.github.io/culebra/).
You can also generate its docs from the source code. Simply change directory
to the `doc` subfolder and type in `make html`, the documentation will be
under `build/html`. You will need [Sphinx] (https://www.sphinx-doc.org/en/master/)
to build the documentation.

---

## 🚀 Quickstart Examples

The `examples` subfolder contains several examples that show the basic usage
of culebra.

---

## 🏛️ Funding

This work is supported by projects:
* *New Computing Paradigms and Heterogeneous Parallel Architectures for
   High-Performance and Energy Efficiency of Classification and Optimization
   Tasks on Biomedical Engineering Applications*
   ([HPEE-COBE] (https://efficomp.ugr.es/research/projects/hpee-cobe/)), with
   reference PGC2018-098813-B-C31
* *Advanced Methods of Biomedical Data Analysis and Brain Modeling Optimized
  for High-Performance and Energy-Efficient Computing*
  ([HPEEC-BIOBRAIN] (https://icarproyectos.ugr.es/msbrainmarker-hpeecbrainmod/)),
  with reference PID2022-137461NB-C31

These projects are funded by the Spanish State Research Agency
MCIN/AEI/[10.13039/501100011033](https://doi.org/10.13039/501100011033) and
by the [European Regional Development Fund (ERDF/EU)](https://ec.europa.eu/regional_policy/en/funding/erdf/).

<div style="text-align: center">
  <a href="https://www.ciencia.gob.es/">
    <img src="https://raw.githubusercontent.com/efficomp/culebra/master/doc/source/_static/micinu.png" height="75">
  </a>
  <a href="https://ec.europa.eu/regional_policy/en/funding/erdf/">
    <img src="https://raw.githubusercontent.com/efficomp/culebra/master/doc/source/_static/erdf.png" height="75">
  </a>
</div>

---

## ⚖️ License

[GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.md).

Culebra © 2020-2026 [EFFICOMP](https://efficomp.ugr.es).
