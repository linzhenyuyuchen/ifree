#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages


with open('README.md', 'r') as readme_file:
    readme = readme_file.read()

requirements = [
    "dicom2nifti",
    "dicompyler-core",
    "lifelines",
    "itk",
    "minepy",
    "numpy",
    "numpy>=1.15.4",
    "pandas>=0.24.2",
    "pyradiomics",
    "scikit-image>=0.14",
    "scipy",
    "SimpleITK",
    "sklearn",
    "tqdm",
    "vtk",
]


setup(
    name='ifree',
    version='0.1.0',
    description="i love freedom, free my hand.",
    long_description=readme,
    long_description_content_type='text/markdown',
    author="Lin Zhenyu",
    author_email='linzhenyu1996@gmail.com',
    url='https://github.com/linzhenyuyuchen/ifree',
    packages=find_packages(include=['ifree']),
    include_package_data=True,
    install_requires=requirements,
    license="BSD license",
    zip_safe=False,
    keywords='ifree',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.6',
    ],
    python_requires='>=3.6'
)

