# Culebra

Culebra is a [DEAP](https://deap.readthedocs.io/en/master/)-based evolutionary computation library designed to solve feature selection problems.

It provides several individual representations, such as bitvectors and sets of feature indices, several fitness functions and several wrapper algorithms.

Experiments and experiment batchs are automatized by means of the *Experiment* and *Batch* classes, both constructed from a [toml](https://pypi.org/project/toml/) configuration file.

## Requirements

Culebra requires Python 3. It also depends on the following Python packages:

* [NumPy](https://numpy.org/doc/stable/)
* [Pandas](https://pandas.pydata.org/docs/)
* [Scikit-learn](https://scikit-learn.org/stable/)
* [DEAP](https://deap.readthedocs.io/en/master/)
* [Toml](https://pypi.org/project/toml/)
* [OpenPyXL](https://openpyxl.readthedocs.io/en/stable/)
* [Sphinx](https://www.sphinx-doc.org/en/master/)

## Documentation

Culebra is fully documented in its [github-pages](https://efficomp.github.io/culebra/). You can also generate its docs from the source code. Simply change directory to the `doc` subfolder and type in `make html`, the documentation will be under `build/html`. You will need [Sphinx](https://www.sphinx-doc.org/en/master/) to build the documentation.

## Usage

The `tests` subfolder contains several examples that show the basic usage of culebra.

## Acknowledgments

This work was supported by project *New Computing Paradigms and Heterogeneous Parallel Architectures for High-Performance and Energy Efficiency of Classification and Optimization Tasks on Biomedical Engineering Applications* ([HPEE-COBE](https://atcproyectos.ugr.es/efficomp/research/projects/hpee-cobe/)), with reference PGC2018-098813-B-C31, funded by the Spanish [*Ministerio de Ciencia, Innovación y Universidades*](https://www.ciencia.gob.es/), and by the [European Regional Development Fund (ERDF)](https://ec.europa.eu/regional_policy/en/funding/erdf/).

<div style="text-align: right">
  <a href="https://www.ciencia.gob.es/">
    <img src="https://raw.githubusercontent.com/efficomp/culebra/master/doc/source/_static/micinu.png" height="75">
  </a>
  <a href="https://ec.europa.eu/regional_policy/en/funding/erdf/">
    <img src="https://raw.githubusercontent.com/efficomp/culebra/master/doc/source/_static/erdf.png" height="75">
  </a>
</div>


## License

[GPLv3](https://www.gnu.org/licenses/gpl-3.0.md) © 2020-2021 [EFFICOMP](https://atcproyectos.ugr.es/efficomp/).