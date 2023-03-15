# thermal_radiation01

This is a repo related to the following work:

> Bach, Y. P. & Ishiguro, M. (2021) "Thermal radiation pressure as a possible mechanism for losing small particles on asteroids", A&A, 654, 113.

[ADS](https://ui.adsabs.harvard.edu/abs/2021A%26A...654A.113B/abstract)

**Citation**:
```
@ARTICLE{2021A&A...654A.113B,
       author = {{Bach}, Yoonsoo P. and {Ishiguro}, Masateru},
        title = "{Thermal radiation pressure as a possible mechanism for losing small particles on asteroids}",
      journal = {\aap},
     keywords = {minor planets, asteroids: general, asteroids: individual: 3200 Phaethon, meteorites, meteors, meteoroids, interplanetary medium, Astrophysics - Earth and Planetary Astrophysics},
         year = 2021,
        month = oct,
       volume = {654},
          eid = {A113},
        pages = {A113},
          doi = {10.1051/0004-6361/202040151},
archivePrefix = {arXiv},
       eprint = {2108.03898},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021A&A...654A.113B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

I also thank Toshi Kasuga (at NAOJ as of Mar 2023) for checking the reproducibility of the notebooks. 

## Contents

**nbviewer [here](https://nbviewer.jupyter.org/github/ysbach/thermal_radiation01/tree/master/).**

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
2. via figshare (doi: 10.6084/m9.figshare.12044883)

For the first option, you are free to go ahead. For the second option, download it, unzip it, and put two ``*.nc`` files (except thos indicated by ``bck``, which means "backup") to ``data/`` directory.


### NOTE
You may encounter ``NameError: name 'InteractiveShell' is not defined``. Please add these lines at the top of the notebooks to avoid this:

```python
# %matplotlib notebook
%config InlineBackend.figure_format = 'retina'
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'last_expr'
```

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
* astroquery
  * ``gh repo clone astropy/astroquery && cd astroquery && pip install -e .``

You may make a virture environment using, e.g., conda, ``conda create -n thermal_ana2021 matplotlib scipy numpy pandas astropy xarray netcdf4 numba`` and then ``conda activate thermal_ana2021``. You may have to specify versions of packages.





## Timeline

Dates below are in Korean Standard Time.

* Idea contrived in circa 2017.
* First conference presentation at [IDP 2019](http://www.perc.it-chiba.ac.jp/meetings/IDP2019/Scientific_Program.html) (2019-02-13)

* **Nature Astronomy**:
  * 2020-04-21: Submitted
  * 2020-04-23: Editor assigned
  * 2020-04-27: Editor declined the paper

* **Nature Communications**:
  * 2021-05-05: Submitted (transferred from NatAs)
  * 2021-05-12: Editor assigned
  * 2021-05-21: All (three) reviewers assigned
  * 2020-06-08: Review completed (1 recommended publication, 2 rejected)
    * Decision: REJECTION

* **Astronomy & Astrophysics**:
  * 2020-12-28: Initial submission
  * 2021-01-11: Draft sent to referee
  * 2021-02-21~2021-07-03: **Review**
    * 2021-02-21: First review letter received
    * 2021-06-02: Second submission
    * 2021-07-21: Second review letter received
    * 2021-07-23: Third submission
    * 2021-07-23: Accepted for publication
  * 2021-08-03~2021-08-05: Language editor
    * 2021-08-03: First comment received
    * 2021-08-03: Re-submitted with correction
    * 2021-08-05: Second comment received
    * 2021-08-06: Re-submitted with correction
  * 2021-08-09: v1 available from [arXiv (2108.03898)](https://arxiv.org/abs/2108.03898)
  * 2021-10-06: Final proof
    * 2021-10-06: Proof (publisher form) received with "Author Query"
    * 2021-10-08: Answers to the query and requested minor corrections (typos).
    * 2021-10-15: Additional simple correction
    * 2021-10-21: Final publication.
