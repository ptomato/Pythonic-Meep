from distutils.core import setup, Extension

ext = Extension(name='_meep',
    sources=['src/meepmodule.cpp', 'src/meepgridvolume.cpp', 
        'src/meepcontinuoussrctime.cpp', 'src/meepboundaryregion.cpp', 
        'src/meepstructure.cpp', 'src/meepfields.cpp', 'src/material.cpp'],
    libraries=['meep'],
    depends=['src/meepmodule.hpp', 'src/material.hpp'])

setup(name='meep',
    py_modules=['meep'],
    ext_modules=[ext])
