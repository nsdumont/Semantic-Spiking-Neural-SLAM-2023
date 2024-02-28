import os
from setuptools import setup, find_packages



setup(
        name = 'sspslam',
        version = '0.1',
        author='Nicole Dumont', 
        author_email='ns2dumont@uwaterloo.ca',
        description=('Code for path intergration, mapping, and memory with Spatial Semantic Pointers'),
        license = 'TBD',
        keywords = '',
        url='http://github.com/', 
        packages=find_packages(),
        classifiers=[
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Intended Audience :: Science/Research'
            ],
        install_requires=[
            'numpy>=3.1.0',
            'scipy',
            'nengo',
            'nengo_spa',
            'mypy'
            ],
)
