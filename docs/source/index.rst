.. Neuronal Cascades documentation master file, created by
   sphinx-quickstart on Sat May  1 14:08:21 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Neuronal Cascades's documentation!
=============================================

``Neuronal Cascades`` is a python package for simulating spreading processes, such as *Watts-Thresholds model* [1,2] or *Simplicial Threshold model* [3] for cascades in which a vertex :math:`v_{i}` becomes active only when the activity across its simplicial neighbors —of which their are different types— surpasses a threshold :math:`T_{i}` over *noisy geometric simplicial complexes*. See the paper [3] for details.

:ref: [1] - Watts, Duncan J.  A simple model of global cascades on random networks, PNAS, 99, 9, 2002, 10.1073/pnas.082090499.
:ref: [2] - Taylor, D., Klimm, F., Harrington, H. et al. Topological data analysis of contagion maps for examining spreading processes on networks. Nature Communications, 6, 7723 (2015). https://doi.org/10.1038/ncomms8723
:ref: [3] - Kilic, B.Ü., Taylor, D. Simplicial cascades are orchestrated by the multidimensional geometry of neuronal complexes. Communications Physics, 5, 278 (2022). https://doi.org/10.1038/s42005-022-01062-3

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Introduction
   Tutorial
   Neuronal Cascades

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
