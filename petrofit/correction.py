from copy import copy

import numpy as np

from scipy.interpolate import interp1d
from scipy.special import gammaincinv

from astropy.utils.console import ProgressBar
from astropy.modeling import models
from astropy import table

from petrofit.modeling.models import PSFConvolvedModel2D, sersic_enclosed, sersic_enclosed_inv
from petrofit.photometry import photometry_step
from petrofit.modeling.fitting import model_to_image
from petrofit.petrosian import Petrosian, calculate_petrosian_r

from matplotlib import pyplot as plt

__all__ = ['generate_petrosian_sersic_correction']


def _generate_petrosian_correction(args):
    r_eff, n, psf, oversample, plot = args
    amplitude = 100 / np.exp(gammaincinv(2. * n, 0.5))

    # Total flux
    r_100 = sersic_enclosed(
        np.inf,
        amplitude=amplitude,
        r_eff=r_eff,
        n=n)
    total_flux = r_100 * 0.99

    # Calculate radii
    r_20, r_80, r_total_flux = [sersic_enclosed_inv(
        total_flux * fraction,
        amplitude=amplitude,
        r_eff=r_eff,
        n=n) for fraction in [0.2, 0.8, 1.0]]

    # Make r_list
    max_r = r_total_flux * 2 if n < 2 else r_total_flux * 1.3
    if max_r >= 200:
        r_list = [x for x in range(1, 201, 2)]
        r_list += [x for x in range(300, int(max_r) + 100, 100)]
    else:
        r_list = [x for x in range(1, int(max_r) + 2, 2)]
    r_list = np.array(r_list)

    image_size = max(r_list) * 2

    x_0 = image_size // 2
    y_0 = image_size // 2

    # Make Model Image
    # ----------------
    # Define model
    galaxy_model = models.Sersic2D(
        amplitude=amplitude,
        r_eff=r_eff,
        n=n,
        x_0=x_0,
        y_0=y_0,
        ellip=0.,
        theta=0.,
    )

    # PSF weap
    galaxy_model = PSFConvolvedModel2D(galaxy_model, psf=psf, oversample=oversample)

    galaxy_image = model_to_image(galaxy_model, image_size, center=(x_0, y_0))

    flux_list, area_list, err = photometry_step((image_size // 2, image_size // 2), r_list, galaxy_image,
                                                plot=plot,
                                                vmax=amplitude / 100)
    # Calculate Photometry and petrosian
    # ----------------------------------
    # Petrosian from Photometry
    p = Petrosian(r_list, area_list, flux_list)
    rc1, rc2, c_index = p.concentration_index()
    if np.any(np.isnan(np.array([rc1, rc2, c_index]))):
        raise Exception("concentration_index cannot be computed (n={}, r_e={})".format(n, r_eff))

    # Compute new r_total_flux
    _, indices = np.unique(flux_list, return_index=True)
    indices = np.array(indices)
    f = interp1d(flux_list[indices], r_list[indices], kind='linear')
    model_r_total_flux = f(total_flux)

    # Compute new r_80
    model_r_80 = f(total_flux * 0.8)

    # Compute corrections
    corrected_epsilon = model_r_total_flux / p.r_petrosian
    corrected_epsilon80 = model_r_80 / p.r_petrosian

    corrected_p = copy(p)
    corrected_p.epsilon = corrected_epsilon

    # Make output list
    # ----------------
    # Petrosian indices
    p02, p03, p04, p05 = [calculate_petrosian_r(p.r_list, p.area_list, p.flux_list, i) for i in (0.2, 0.3, 0.4, 0.5)]
    assert np.round(p.r_petrosian, 6) == np.round(p02, 6)

    u_r_20 = p.fraction_flux_to_r(0.2)
    u_r_80 = p.fraction_flux_to_r(0.8)
    c_r_20 = corrected_p.fraction_flux_to_r(0.2)
    c_r_80 = corrected_p.fraction_flux_to_r(0.8)

    row = [n, r_eff, r_20, r_80, r_total_flux, r_100,
           p02, p03, p04, p05, 5 * np.log(p02 / p05), 5 * np.log(p02 / p03),
           p.epsilon, u_r_80 / p.r_petrosian, p.r_total_flux, u_r_20, u_r_80, p.c2080,
           corrected_epsilon, corrected_epsilon80, corrected_p.r_total_flux, c_r_20, c_r_80, corrected_p.c2080, ]
    if plot:
        corrected_p.plot(True, True)
        plt.show()
        print(corrected_epsilon)
        print(r_eff, p.r_half_light, corrected_p.r_half_light)
        print(" ")

    del galaxy_model, galaxy_image
    del flux_list, area_list, err
    del corrected_p, p

    return row


def generate_petrosian_sersic_correction(output_yaml_name, psf=None, r_eff_list=None, n_list=None,
                                         oversample=('x_0', 'y_0', 10, 10), out_format=None, overwrite=False,
                                         ipython_widget=True, n_cpu=None, plot=True):
    """
    Generate corrections for Petrosian profiles by simulating a galaxy image (single component sersic) and measuring its
    properties. This is done to identify the correct `epsilon` value that, when multiplied with `r_petrosian`, gives
    `r_total_flux`. To achieve this, an image is created from a Sersic model and convolved with a PSF (if provided).
    The Petrosian radii and concentrations are computed using the default `epsilon` = 2. Since the real `r_total_flux`
    of the simulated galaxy is known, the correct `epsilon` can be determined by
    `epsilon = r_petrosian / corrceted_r_total_flux`. The resulting grid is used to map measured properties to the correct
    `epsilon` value. If `output_yaml_name` is provided, the grid is saved to using an astropy table file which is readable
    by `petrofit.petrosian.PetrosianCorrection`.

    Parameters
    ----------
    output_yaml_name : str
        Name of output file, must have .yaml or .yml extension.

    psf : numpy.array or None
        2D PSF image to pass to `petrofit.fitting.models.PSFConvolvedModel2D`.

    r_eff_list : list, (optional)
        List of `r_eff` (half light radii) in pixels to evaluate.

    n_list : list, (optional)
        List of Sersic indices to evaluate.

    oversample : int or tuple
        oversampling to pass to `petrofit.fitting.models.PSFConvolvedModel2D`.

    out_format : str, optional
        Format passed to the resulting astropy table when writing to file.

    overwrite : bool, optional
        Overwrite if file exists.

    ipython_widget : bool, optional
        If True, the progress bar will display as an IPython notebook widget.

    n_cpu : bool, int, optional
        If True, use the multiprocessing module to distribute each task to a different
        processor core. If a number greater than 1, then use that number of cores. This
        should be selected taking ram in consideration (since high n and large r_eff
        create large images).

    plot : bool
        Shows plot of photometry and Petrosian. Not available if n_cpu > 1.

    Returns
    -------
    petrosian_grid : Table
        Astropy Table that is readable by `petrofit.petrosian.PetrosianCorrection`
    """

    if r_eff_list is None:
        r_eff_list = np.arange(10, 100 + 5, 5)

    if n_list is None:
        n_list = np.arange(0.5, 6.0 + 0.5, 0.5)

    args = []
    for n_idx, n in enumerate(n_list):
        for r_eff_idx, r_eff in enumerate(r_eff_list):
            args.append([r_eff, n, psf, oversample, plot])

    if n_cpu is None or n_cpu == 1:
        with ProgressBar(len(args), ipython_widget=ipython_widget) as bar:
            rows = []
            for arg in args:
                row = _generate_petrosian_correction(arg)
                rows.append(row)
                bar.update()
    else:
        assert plot == False, 'Plotting not available for ncpu > 1'
        step = 100 if len(r_eff_list) * len(n_list) > 500 else 2
        rows = ProgressBar.map(_generate_petrosian_correction, args, multiprocess=n_cpu,
                               ipython_widget=ipython_widget, step=step)

    names = ['n', 'r_eff', 'sersic_r_20', 'sersic_r_80', 'sersic_r_99', 'sersic_r_100',
             'p02', 'p03', 'p04', 'p05', 'p0502', 'p0302',
             'u_epsilon', 'u_epsilon_80', 'u_r_99', 'u_r_20', 'u_r_80', 'u_c2080',
             'c_epsilon', 'c_epsilon_80', 'c_r_99', 'c_r_20', 'c_r_80', 'c_c2080', ]
    petrosian_grid = table.Table(rows=rows, names=names)

    if output_yaml_name is not None:
        try:
            petrosian_grid.write(output_yaml_name, format=out_format, overwrite=overwrite)
        except Exception as e:
            print('Could not save to file: {}'.format(e))
            print('You can save the returned table using `petrosian_grid.write`')
    return petrosian_grid