# Neuronal_Cascades

``Neuronal Cascades`` is the python module that runs simplicial contagion maps on custom build geometric networks with heterogeneous node properties. In a nutshell, we want to investigate the dynamics of a simplicial contagion starting from a seed cluster and spreading across the underlying network according to a stochastic threshold model. The model and hence the package is as general as possible in a way that one can play with the parameters to obtain different network topologies and contagion models.

## Paper

Kilic, B.Ü., Taylor, D. Simplicial cascades are orchestrated by the multidimensional geometry of neuronal complexes. Communications Physics, 5, 278 (2022). https://doi.org/10.1038/s42005-022-01062-3

## Installation/Usage

As the package has not been published on PyPi yet, it CANNOT be installed using ``pip``. ``Neuronal Cascades`` uses cython to exploit the faster computation compared to python. So, you have to ``setup.py`` the ``*.pyx`` files in ``Neuronal_Cascades_base``.

For now, the suggested method is having the same folder directory in Neuronal_Cascades/Simplicial_Model (setup.py has to be in the parent directory of Neuronal_Cascades_base). Then, add an empty ``__init__.py`` file into Neuronal_Cascades_base (Github doesn't allow adding empty files). Then go to this directory on your terminal and run ``python setup.py build_ext --inplace``. See [Cython](https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html)
for details.

## Documentation

[Documentation](https://neuronal-cascades.readthedocs.io/en/latest/index.html) available for this python package.


