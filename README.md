# thermal_radiation01

**nbviewer [here](https://nbviewer.jupyter.org/github/ysbach/thermal_radiation01/tree/master/).**

This is a repo related to the following work:

> Bach, Y. P. & Ishiguro, M. (2021) "Thermal radiation pressure as a possible mechanism for losing small particles on asteroids", A&A, under review



## Contents

| Notebook                 | Contents                                                     | Related figures |
| ------------------------ | ------------------------------------------------------------ | --------------- |
| 01-Qpr_Calculation.ipynb | Calculation of $ Q_\mathrm{pr} $​ values. The calcaulted values are saved to ``data/`` directory (available from this repo) | 2, A1, A2       |
| 02-TPM_Check.ipynb       | Validation of our TPM code against a previous work (Mueller M. 2007 dissertation) and some exploration of parameterization. |                 |
| 03-acc_eq_time.ipynb     | Accelerations on equator of model asteroids.                 | 3, 4, 5, 9      |
| 04-accplot.ipynb         | Diagrams related to the accelerations. The calculation requires huge amount of time (see the note below) | 6, 7, 8         |
| 05-Phaethon_Orbit.ipynb  | Calculations related to Phaethon's orbit.                    | 11              |
| 06-Theta_rh.ipynb        | Thermal parameter and heliocentric distance relation.        | 10              |
| 07-trajectory.ipynb      | Particle trajectory calculations.                            | 12, 13          |

* ``yssbtmpy_ana2021``: Snapshot of the under-development package, ``yssbtmpy`` used for this work.
* ``accutil.py``: Utility scripts. (``accutil_archive.py`` is saved purely for legacy purpose - please don't use it.)
* ``eph17.csv``, ``phae_ephem.csv``: Cached ephemerides to plot in ``06-Theta_rh.ipynb``.



### NOTE

At this stage, the output data from ``04-accplot.ipynb`` (all the ``.nc`` files) are available either

1. you can make it from the code at ``04-accplot.ipynb``, or
2. via figshare (doi: 10.6084/m9.figshare.12044883) - **not published yet -- will be published as soon as the paper is accepted**.

For the first option, you are free to go ahead. For the second option, download it, unzip it, and put two ``*.nc`` files (except thos indicated by ``bck``, which means "backup") to ``data/`` directory.



### Requirements

* python 3.6+
* matplotlib
* scipy
* numpy 
* pandas 
* astropy
* xarray
  * install netcdf4 by ``conda install -c conda-forge netcdf4 --yes``
* numba
  * ``yssbtmpy_ana2021`` requires numba for TPM calculation.

You may make a virture environment using, e.g., conda, ``conda create -n thermal_ana2021 matplotlib scipy numpy pandas astropy xarray netcdf4 numba`` and then ``conda activate thermal_ana2021``. You may have to specify versions of packages.





## Timeline

* Idea contrived in circa 2017. 

* First conference presentation at [IDP 2019](http://www.perc.it-chiba.ac.jp/meetings/IDP2019/Scientific_Program.html) (2019-02-13)

Nature Astronomy:

* 2020-04-21: Submitted
* 2020-04-23: Editor assigned
* 2020-04-27: Editor declined the paper

Nature Communications:

* 2021-05-05: Submitted (transferred from NatAs)
* 2021-05-12: Editor assigned
* 2021-05-21: All (three) reviewers assigned
* 2020-06-08: Review completed (1 recommended publication, 2 rejected)

Astronomy & Astrophysics:

* 2020-12-28: Initial submission
* 2021-01-11: Draft sent to referee
* 2021-02-21: First review letter received
* 2021-06-02: Second submission
* 2021-07-21: Second review letter received
* 2021-07-23: Third submission
* 2021-07-23: Accepted for publication
* 
