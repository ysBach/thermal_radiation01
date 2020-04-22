import yssbtmpy as tm
from matplotlib import pyplot as plt
from matplotlib import rcParams
from astropy.visualization import ZScaleInterval, ImageNormalize
import numpy as np


def znorm(image):
    return ImageNormalize(image, interval=ZScaleInterval())


def zimshow(ax, image, **kwargs):
    return ax.imshow(image, norm=znorm(image), origin='lower', **kwargs)


plt.style.use('default')
rcParams.update({'font.size': 12})

if __name__ == "__main__":
    _, a_bond, _ = tm.solve_pAG(p_vis=0.1, slope_par=0.1)

    sb = tm.SmallBody()
    ths = [0, 10]
    # fig, axs = plt.subplots(2, len(ths), figsize=(6*len(ths), 6))
    # for i, th in enumerate(ths):
    #     sb.minimal_set(th, 30)
    #     sb.set_tpm()
    #     sb.calc_temp()
    #     print(sb.mu_suns)
    #     print(sb.tempsurf)
    #     zimshow(axs[0, i], sb.mu_suns)
    #     zimshow(axs[1, i], sb.tempsurf.value, vmin=0.7, vmax=sb.tempsurf.value.max())
    # plt.show()

    # hdul = sb.tohdul(output='test.fits', overwrite=True)
    # print(hdul[0].header)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for i, th in enumerate(ths):
        sb.minimal_set(th, 0.01)
        sb.set_tpm(nlat=90)
        sb.calc_temp()
        print(sb.mu_suns.shape)
        print(sb.tempsurf.shape)

        ax.plot(sb.tempsurf[0, :], label=f"th = {th:.3f}")

    hdul = sb.tohdul(output='test.fits', overwrite=True)
    print(hdul[0].header)

    ax.legend()
    plt.show()
    # Gpar = 0.15
    # p_V = tm.solve_pAG(A_Bond=0.1, Gpar=Gpar)
    # for k in range(5):
    #     for ti in [0, 1, 10, 100, 1000]:
    #         phys = dict(
    #             ti=ti,
    #             epsilon=0.9,
    #             period_rot=3.6,
    #             p_vis=p_V,
    #             slope_par=Gpar,
    #             slon__deg=0,
    #             slat__deg=90
    #             )
    #         ephem = dict(
    #             hlon__deg=0,
    #             hlat__deg=0,
    #             rh__au=1.1
    #         )
    #         sb = tm.SmallBody(**phys)
    #         tb = tm.TPMBody(**phys, **ephem, nZ=50)
    #         tb.calc_temp()
    #         if k == 1:
    #             print(tb.tempsurf[1, 2])
    # hdul = tb.tohdul(output=None)
    # print(hdul[0].header)
    # print(hdul[1].data)
