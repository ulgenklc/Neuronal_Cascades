#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("Neuronal_Cascades_base.Geometric_Brain_Network", ["Neuronal_Cascades_base/Geometric_Brain_Network.pyx"]
),
]

for e in extensions:
    e.cython_directives = {"embedsignature": True}

setup(
    name = "Neuronal_Cascades_base",
    ext_modules = cythonize(extensions, compiler_directives={'language_level' : "3"}),
    include_dirs=[numpy.get_include()],
    zip_safe = False,
)

