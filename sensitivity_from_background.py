"""
This script calculates the sensitivity requirements of PAHST using two methods:
1. Telescope background emission model from JWST: Swinyard et al. (2004) 
2. Based on NEP of detector, which assumes no background emission.

Sources are provided where possible. 

It is useful to know the definitions of radiant power, radiant flux, 
spectral radiance, spectral irradiance etc:
https://en.wikipedia.org/wiki/Radiant_flux
"""


import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const

import pandas as pd

def planck_lam(wave, T):
	"""
	Black body spectrum as function of wavelength.

	Parameters
	----------
	wave : Quantity
		Wavelength or astropy.units equivalent.
	T : Quantity
		Temperature of black body.

	Returns
	-------
	 : Quantity
		Spectral radiance [W/sr/m**3] at wavelengths corresponding to wave.
	"""
	wave = wave.to(u.micron, equivalencies=u.spectral())
	return (2*const.h*const.c**2/wave**5/u.sr *
			1/(np.exp(
				(const.h*const.c/wave/const.k_B/T).decompose())-1))\
		.to(u.W/u.sr/u.m**3)

#load spectral resolution
df = pd.read_csv('2019-03-07_PACS_data_from_plot.csv')
lambda_R = np.array(df)

wave = lambda_R[:,0]*u.micron
R = lambda_R[:,1]


# wave = np.linspace(20, 200, 1000)*u.micron
# wave = np.linspace(5, 20, 1000)*u.micron


#### Zodiacal light calculation in Swinyard et al. (2004)
Z_F = 1.2
B_5500 = planck_lam(wave, 5500*u.K)
B_270 = planck_lam(wave, 270*u.K)
# zodiacal light spectral radiance (per steradian)
S_zod = Z_F*(3.5e-14*B_5500 + 3.58e-8*B_270)


#### Calculate the spectral radiance of the telescope background
# based on the model from Swinyard et al. (2004)
#all temperatures
T_detector = 0.05*u.K
T_telescope = 4*u.K
T_sunshield_1 = 35*u.K
T_sunshield_2 = 60*u.K

S_dect_mirror = 0.48*planck_lam(wave, T_detector) + \
		0.1*planck_lam(wave, T_telescope)

S_shield_1 = 3e-5*planck_lam(wave, T_sunshield_1) #original view factor: 3e-5
S_shield_2 = 9.9e-7*planck_lam(wave, T_sunshield_2) #original view factor: 9.9e-7

S_sunshield = S_shield_1 + S_shield_2
S_tel = S_dect_mirror #+ S_sunshield
S_tot = S_zod + S_tel


#### some semi-fixed values
#desired signal to noise
# desired_SN = 10
#FWHM of telescope; galaxy is effectively a point source
fwhm = 1.8 * u.arcsecond
#spectral resolution at 55 microns from Fig 13 of PAHST paper, page 37
# R_55 = 2666.
#telescope diameter
d_tel = 8*u.m


#target signal to noise
SN_goal = 10
#integration time
t_i = 10000*u.s#(20*u.h).to(u.s)
#### Do the reverse calculation: sensitivity from SN

'''
## first determine counts from background
# also calculate the wavelength of this emission
wave_rest_frame = 7.7*u.micron #PAH line
#redshift
z = 6.
wave_redshifted = wave_rest_frame * (z+1)
'''
#now determine the wavelength bin at 7.7 microns redshifted to 54 
#microns
####USE SPECTRAL RESOLUTION WHICH VARIES WITH WAVELENGTH
delta_lambda = wave/R
#convert to frequency bin
delta_f = (const.c/(wave.to(u.m)**2)*delta_lambda.to(u.m)).to(u.Hz)


#calculate telescope area
A_tel = np.pi*(d_tel/2)**2

#convert the telescope FWHM to a solid angle
gal_solid_angle = ((fwhm/2.)**2 * np.pi).to(u.sr)

#radiant flux of the background emission
P_background = S_tot*gal_solid_angle * delta_lambda.to(u.m) * A_tel

#conversion factor for photon wavelength to photon energy
photon_energy_range = const.h*const.c/wave.to(u.m)

#calculate the number of photons per seconds
N_background = P_background.to(u.J/u.s)/photon_energy_range


#now the number of counts of the source is given by the SNR
N_source = (SN_goal*(N_background*t_i)**0.5)/t_i
#convert this number of counts of the source to radiant flux [W]
P_source = (N_source * photon_energy_range).to(u.W)
#convert to flux density [W/m^2]
F_source = P_source/A_tel
#convert to spectral irradiance/spectral flux density [W/m^2/Hz]
S_source = (F_source/delta_f).to(u.Jy)

#for photometer
# Band = np.array([47, 87, 155])*u.micron
# Bandpass = np.array([34, 46, 90])*u.micron


#### plot sensitivity
plt.scatter(wave, S_source*1e6, label = 'Sensivity')
# plt.scatter(wave_redshifted, P_gal*t_i, label = 'Galaxy emission', marker = '*', color = 'r')


#### Beautify the plot
plt.grid(alpha=0.4)
plt.yscale('log')
plt.ylabel('Spectral irradiance/spectral flux density [$\mu Jy$]')
plt.xlabel(r'Wavelength $[\mu m]$')
plt.title(f'PAHST sensivity at S/N = {SN_goal} and {t_i} integration time')

# plt.ylim((1e-10, 1e-2))

plt.legend(loc = 'best')
plt.savefig(f'PAHST_sensivity.png', dpi = 300, bbox_inches = 'tight')
# plt.show()
plt.close()
