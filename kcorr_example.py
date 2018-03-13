# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:07:00 2014
Applying k-corrections to the data.

@author: Seméli Papadogiannakis
"""
from __future__ import division

import numpy as np
import scipy.interpolate as interpolate
import operator

from astLib import astSED

class Passband:
    """This class describes a filter transmission curve. Passband objects are created by loading data from
    from text files containing wavelength in angstroms in the first column, relative transmission efficiency
    in the second column (whitespace delimited). For example, to create a Passband object for the 2MASS J
    filter:

    passband=astSED.Passband("J_2MASS.res")

    where "J_2MASS.res" is a file in the current working directory that describes the filter.
    w
    Wavelength units can be specified as 'angstroms', 'nanometres' or 'microns'; if either of the latter,
    they will be converted to angstroms.

    """
    def __init__(self, filename, normalise = True, inputUnits = 'angstroms'):
        data = np.genfromtxt(filename)

        self.wavelength = data[:,  0]
        self.transmission = data[:, -2]
        if inputUnits == 'angstroms':
            pass
        elif inputUnits == 'nanometres':
            self.wavelength = self.wavelength * 10.0
        elif inputUnits == 'microns':
            self.wavelength = self.wavelength * 10000.0
        elif inputUnits == 'mm':
            self.wavelength = self.wavelength * 1e7
        elif inputUnits == 'GHz':
            self.wavelength = 3e8 / (self.wavelength * 1e9)
            self.wavelength = self.wavelength * 1e10
        else:
            raise Exception("didn't understand passband input units")

        # Sort into ascending order of wavelength otherwise normalisation will be wrong
        merged = np.array([self.wavelength, self.transmission]).transpose()
        sortedMerged = np.array(sorted(merged, key=operator.itemgetter(0)))
        self.wavelength = sortedMerged[:, 0]
        self.transmission = sortedMerged[:, 1]

        # Store a ready-to-go interpolation object to speed calculation of fluxes up
        self.interpolator = interpolate.interp1d(self.wavelength, self.transmission, kind='linear')

class Spectrum(astSED.SED) :
    """
    At the moment we derive from astSED.SED but let's keep in mind
    that we want to be able to base this class on some other library
    in the future since it is not clear what direction the development
    will take.
    """

    def iflux(self,l) :
        """
        Calculate the interpolated flux for the specified wavelength.
        """

        lamb,flux = self.wavelength,self.flux
        ii = interpolate.interp1d(lamb,flux,kind='linear')
        return ii(l)

    def __resampler(self, band, spectrum=None):
            '''
            Return the values of lambda present in both the filter (zero values
            excluded) and the spectrum. Sorted.
            '''
            transmitivity = band.transmission
            lambda_filter = band.wavelength

            # Indexes where the T is positive:
            nonzero = np.nonzero(transmitivity)[0]

            # Interval of lambdas
            l_min = lambda_filter[nonzero[0]]
            l_max = lambda_filter[nonzero[-1]]

            # Get the lambdas from the spectrum:
            if spectrum is None:
                wavels = self.wavelength
            else:
                wavels = spectrum.wavelength
            id1, id2 = np.searchsorted(wavels, [l_min, l_max])
            lambdas = np.unique(np.concatenate((wavels[id1:id2],
                                                lambda_filter[nonzero])))
            lambdas.sort()

            return lambdas


    def __kcorr_1band(self, band, z):
        '''
        K-correction according to eq. 1 in Kim et al. 1996.
        '''
        lambdas = self.__resampler(band)

        try:
            sed_SN = self.iflux(lambdas)
            sed_SN_rf = self.iflux(lambdas/(1+z))
        except ValueError:
            raise ValueError('The spectrum is out of range.')

        transmission_function = interpolate.interp1d(band.wavelength,
                                            band.transmission,kind='linear')
        transmission = transmission_function(lambdas)

        nominator = np.trapz(sed_SN * transmission, lambdas)
        denominator = np.trapz(sed_SN_rf * transmission, lambdas)

        return 2.5 * (np.log10(1 + z) + np.log10(nominator/denominator))
    '''
     def __kcorr_2band(self, band1, band2, z):
        #
        #Generalised k-correction.
        #
        lambdas1 = self.__resampler(band1)
        lambdas2 = self.__resampler(band2)

        transmission_function_1 = interpolate.interp1d(band1.wavelength,
                                            band1.transmission,kind='linear')
        transmission_1 = transmission_function_1(lambdas1)

        transmission_function_2 = interpolate.interp1d(band2.wavelength,
                                            band2.transmission,kind='linear')
        transmission_2 = transmission_function_2(lambdas2)

        try:
            sed_SN = self.iflux(lambdas1)
            sed_SN_rf = self.iflux(lambdas2/(1+z))
        except ValueError:
            raise ValueError('The spectrum is out of range.')

        nominator_spectrum = np.trapz(sed_SN * transmission_1, lambdas1)
        denominator_spectrum = np.trapz(sed_SN_rf * transmission_2, lambdas2)
        spec = nominator_spectrum / denominator_spectrum

        # We shall correct the colour zero-point, using Vega as a reference.
        vega = astSED.VegaSED()
        lambdas1 = self.__resampler(band1, vega)
        lambdas2 = self.__resampler(band2, vega)

        sed_vega_interpol = interpolate.interp1d(vega.wavelength, vega.flux)

        try:
            sed_vega_1 = sed_vega_interpol(lambdas1)
            sed_vega_2 = sed_vega_interpol(lambdas2)
        except ValueError:
            raise ValueError('The spectrum of Vega is out of range.')

        nominator_colour = np.trapz(sed_vega_1 * transmission_function_1(lambdas1),
                                    lambdas1)
        denominator_colour = np.trapz(sed_vega_2 * transmission_function_2(lambdas2),
                                      lambdas2)

        filt_czp = nominator_colour / denominator_colour

        return 2.5 * (np.log10(1 + z) + np.log10(spec) - np.log10(filt_czp))
    '''
    def kcorr(self, band1, band2=None, z=0.0) :
        """
        Calculate the kcorrection from x to y, where x and y are
        of the class Passband.

        If band2 is None, the classic k-correction is applied (eq. 1).
        Otherwise, the Generalized K-correction in accordance with eq. (2) in
        Kim, Goobar and Perlmutter 1996, PASP 108, 190

        It fails gracefully when the spectrum is out of range.
        """
        if band2 is None or type(band2) is int or type(band2) is float:
            return self.__kcorr_1band(band1, z)
        else:
            #return self.__kcorr_2band(band1, band2, z)
            return NotImplementedError


'''
Need to import all the spectra from the PTF+iPTF sample, the Hsiao model
spectrum and the transmissivity of the filter used on the P48 instrument
from which the photometry is used.
'''

r_filter = Passband('filter_passband/PTF48_rfilt.txt')

model_spectrum_data = np.genfromtxt('snflux_1a.dat')

epoch = model_spectrum_data[:, 0]
wavel_spec = model_spectrum_data[:, 1]
intensity_spec = model_spectrum_data[:, 2]


def apply_kcorr(times, z):
    '''
    Given an epoch and redshift returns k-correction.
    Using the Hsiao spectral template and the R-band
    of the P48 instrument.
    '''

    # Import the filter and the model spectra for
    # different epochs.
    r_filter = Passband('filter_passband/PTF48_rfilt.txt')
    model_spectrum_data = np.genfromtxt('snflux_1a.dat')

    epoch = model_spectrum_data[:, 0]
    wavel_spec = model_spectrum_data[:, 1]
    intensity_spec = model_spectrum_data[:, 2]

    kcorr_hsiao = []

    # Compute k-corrections for each epoch for a
    # specific redshift.
    for day in np.unique(epoch):
        if day == -20: continue
        where = epoch == day
        itsy = intensity_spec[where]
        spectrum = Spectrum(wavel_spec[where], itsy)
        kcorr_hsiao.append(spectrum.kcorr(r_filter, z=z))

    dates = np.arange(-19, 85)
    kcorr_smooth = interpolate.interp1d(np.unique(epoch)[1:], kcorr_hsiao, kind ='linear', bounds_error=False, fill_value=0)

    return kcorr_smooth(times)


# Only plot if this script is run alone.
if __name__ == '__main__':

    # Here I am applying a K-correction for day 1 and z=0.2
    # for a type Ia SN:
    print apply_kcorr(20.0, 0.5)
