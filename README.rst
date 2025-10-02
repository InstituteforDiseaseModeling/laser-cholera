========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - |github-actions| |codecov|
    * - package
      - |version| |wheel| |supported-versions| |supported-implementations| |commits-since|
.. |docs| image:: https://readthedocs.org/projects/laser-cholera/badge/?style=flat
    :target: https://laser-cholera.readthedocs.io/en/latest/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/InstituteforDiseaseModeling/laser-cholera/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/InstituteforDiseaseModeling/laser-cholera/actions

.. |codecov| image:: https://codecov.io/gh/InstituteforDiseaseModeling/laser-cholera/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://app.codecov.io/github/InstituteforDiseaseModeling/laser-cholera

.. |version| image:: https://img.shields.io/pypi/v/laser-cholera.svg
    :alt: PyPI Package latest release
    :target: https://test.pypi.org/project/laser-cholera/

.. |wheel| image:: https://img.shields.io/pypi/wheel/laser-cholera.svg
    :alt: PyPI Wheel
    :target: https://test.pypi.org/project/laser-cholera/

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/laser-cholera.svg
    :alt: Supported versions
    :target: https://test.pypi.org/project/laser-cholera/

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/laser-cholera.svg
    :alt: Supported implementations
    :target: https://test.pypi.org/project/laser-cholera/

.. |commits-since| image:: https://img.shields.io/github/commits-since/InstituteforDiseaseModeling/laser-cholera/v0.9.1.svg
    :alt: Commits since latest release
    :target: https://github.com/InstituteforDiseaseModeling/laser-cholera/compare/v0.9.1...main



.. end-badges

LASIK - LASER based SImulation of Kolera

* Free software: MIT license

Installation
============

::

    pip install laser-cholera

You can also install the in-development version with::

    pip install https://github.com/InstituteforDiseaseModeling/laser-cholera/archive/main.zip


Documentation
=============


https://laser-cholera.readthedocs.io/en/latest/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
