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
import matplotlib
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
#telescope diameter
d_tel = 8*u.m #DEFAULT PAHST = 8m
#FWHM of telescope; galaxy is effectively a point source
fwhm = ((1.22 * wave.to(u.m)/d_tel)*u.rad).to(u.arcsecond)
optical_efficiency = 0.4

#target signal to noise
SN_goal = 5**2 #sigma = 5
#integration time
t_i = 3600*u.s#(20*u.h).to(u.s)
#### Do the reverse calculation: sensitivity from SN
#now determine the wavelength bin
delta_lambda = wave/R
#convert to frequency bin
delta_f = (const.c/(wave.to(u.m)**2)*delta_lambda.to(u.m)).to(u.Hz)


#calculate telescope area
A_tel = np.pi*(d_tel/2)**2

#convert the telescope FWHM to a solid angle
gal_solid_angle = ((fwhm/2.)**2 * np.pi).to(u.sr)

#radiant flux of the background emission
P_background = S_tot*gal_solid_angle * delta_lambda.to(u.m) * A_tel #should be delta_lambda

#conversion factor for photon wavelength to photon energy
photon_energy_range = const.h*const.c/wave.to(u.m)

#calculate the number of photons per seconds
N_background = optical_efficiency*P_background.to(u.J/u.s)/photon_energy_range


#now the number of counts of the source is given by the SNR
N_source = ((SN_goal*(N_background*t_i)**0.5)/t_i)/optical_efficiency
#convert this number of counts of the source to radiant flux [W]
P_source = (N_source * photon_energy_range).to(u.W)
#convert to flux density [W/m^2]
F_source = P_source/A_tel
#convert to spectral irradiance/spectral flux density [W/m^2/Hz]
S_source = (F_source/delta_f).to(u.Jy)

#for photometer
# Band = np.array([47, 87, 155])*u.micron
# Bandpass = np.array([34, 46, 90])*u.micron

def sensitivity_other_missions():
	"""
	Plot the sensitivity of other instruments out there
	"""

	# transparency
	aa = 0.8

	###
	n_inst = 12
	colors = plt.cm.viridis(np.linspace(0,1,n_inst))

	plt.figure(figsize=(8,5))

	llines = []

	### SPICA
	# SMI: http://www.spica-mission.org/downloads/smi-fs.pdf
	mrs_ll = np.arange(18,36,0.1) *u.micrometer
	mrs_sy = [3e-20,20e-20]	*u.W/u.m**2		# 1 hr, 5 Sigma
	plt.fill_between(mrs_ll, np.repeat(mrs_sy[0], len(mrs_ll)), np.repeat(mrs_sy[1], len(mrs_ll)), \
		alpha=aa/2, label='SPICA, SMI MR', \
		color=colors[0])
	#
	hrs_ll = np.arange(12,18,0.1) *u.micrometer
	hrs_sy = [1.5e-20, 2e-20] *u.W/u.m**2	# 1 hr, 5 Sigma
	plt.fill_between(hrs_ll, np.repeat(hrs_sy[0], len(hrs_ll)), np.repeat(hrs_sy[1], len(hrs_ll)), \
		alpha=aa/2, label='SPICA, SMI HR', \
		color=colors[1])
	#
	# SAFARI: http://www.spica-mission.org/downloads/safari-fs.pdf
	safari_ll = [45., 72., 115., 185.] *u.micrometer
	safari_sy = [7.2e-20, 6.6e-20, 6.6e-20, 8.2e-20] *u.W/u.m**2		# 1 hr, 5 Sigma
	l2, = plt.plot(safari_ll, safari_sy, ls='', marker='o', \
		alpha=aa, label='SPICA, Safari', \
		color=colors[2])
	llines.append(l2)


	### JWST
	# MIRI: http://iopscience.iop.org/article/10.1086/682252/pdf Table 2
	miri_ll = np.array([5.6, 7.7, 10, 11.3, 12.8, 15, 18, 21, 25.5]) *u.micrometer
	miri_R = np.array([5, 3.5, 5, 16, 5, 5, 6, 4, 6])
	miri_delta_ll = miri_ll/miri_R
	miri_delta_nu = const.c / (miri_ll.to(u.meter)**2) * miri_delta_ll.to(u.m)
	miri_sy = [0.16, 0.25, 0.56, 1.35, 0.84, 1.39, 3.46, 7.09, 26.2] * u.microjansky	# 10,000 s, 10 sigma 
	miri_sy_wm2 = (miri_sy * miri_delta_nu).to(u.W/u.m**2)
	l3, = plt.plot(miri_ll, miri_sy_wm2, ls='', marker='s', \
		alpha=aa, label='JWST, MIRI', \
		color=colors[3])

	llines.append(l3)
	#
	# NIRCam: https://jwst-docs.stsci.edu/display/JTI/NIRCam+Filters Tables 2 and 3
	ncam_ll_w = [0.7, 0.90, 1.15, 1.50, 2.00, 2.77, 3.22, 3.56, 4.44] *u.micrometer
	#### NEEDS TO BE CHANGED TO DELTA NU
	ncam_nu_w = const.c / ncam_ll_w.to(u.meter)
	ncam_sy_w = [22.5, 15.3, 13.2, 10.6, 9.1, 16.3, 9.1, 12.1, 23.6] *u.nanojansky
	l4, = plt.plot(ncam_ll_w, (ncam_sy_w*ncam_nu_w).to(u.W/u.m**2), marker='s', ls='', \
		alpha=aa, color = colors[4], label='JWST, NIRCam W')
	llines.append(l4)
	#
	ncam_ll_m = [1.40, 1.62, 1.82, 2.10, 2.50, 3.00, 3.55, 3.60, 4.10, 4.30, 4.60, 4.80] *u.micrometer
	ncam_nu_m = const.c / ncam_ll_m.to(u.meter)
	ncam_sy_m = [19.4, 21.4, 16.1,  14.9, 32.1, 25.8, 21.8, 20.7, 24.7, 50.9, 56.5, 67.9] *u.nanojansky
	l5, = plt.plot(ncam_ll_m, (ncam_sy_m*ncam_nu_m).to(u.W/u.m**2), ls='', marker='s', \
		alpha=aa, color = colors[5], label='JWST, NIRCam M')
	llines.append(l5)
	#
	ncam_ll_n = [1.64, 1.87, 2.12, 3.23, 4.05, 4.66, 4.70] *u.micrometer
	ncam_nu_n = const.c / ncam_ll_n.to(u.meter)
	ncam_sy_n = [145, 133, 129, 194, 158, 274, 302] *u.nanojansky
	l6, = plt.plot(ncam_ll_n, (ncam_sy_n*ncam_nu_n).to(u.W/u.m**2), ls='', marker='s', \
		alpha=aa, color = colors[6], label='JWST, NIRCam N')
	llines.append(l6)


	### Herschel
	# SPIRE: http://herschel.esac.esa.int/Docs/SPIRE/html/spire_om.html#x1-690004.2.3 Table 4.3
	spire_ll_ssw =  [194, 214, 282] *u.micrometer
	spire_sy_ssw =  [2.15e-17, 1.56e-17, 1.56e-17] *u.W/u.m**2
	l7, = plt.plot(spire_ll_ssw, spire_sy_ssw, ls='', marker='o', \
		alpha=aa, color = colors[7], label='Herschel, SPIRE SSW')
	llines.append(l7)
	#
	spire_ll_slw =  [313, 392, 671] *u.micrometer
	spire_sy_slw =  [2.04e-17, 0.94e-17, 2.94e-17] *u.W/u.m**2
	l8, = plt.plot(spire_ll_slw, spire_sy_slw, ls='', marker='o', \
		alpha=aa, color = colors[8], label='Herschel, SPIRE SLW')
	llines.append(l8)
	#
	# PACS: https://www.cosmos.esa.int/documents/12133/996891/PACS+Spectrometer+Fact+Sheet
	pacs_ll =  [70, 92, 140, 180] *u.micrometer
	pacs_sy =  [7.2e-18, 6.8e-18, 2.4e-18, 3.6e-18] *u.W/u.m**2
	l9, = plt.plot(pacs_ll, pacs_sy, ls='', marker='o', \
		alpha=aa, color = colors[9], label='Herschel, PACS spectr.')
	llines.append(l9)

	'''
	### PAHST
	# imager
	band = np.array([47, 87, 155])*u.micron
	l10, = plt.plot(band, f_PAHST_imager, ls='', marker='o', \
		alpha=aa, color = colors[10], label='PAHST, imager')
	llines.append(l10)
	# spectrometer
	wave = np.array([47, 87, 155])*u.micron
	l11, = plt.plot(wave, f_PAHST_spectrometer, ls='', marker='o', \
		alpha=aa, color = colors[11], label='PAHST, spectrometer')
	llines.append(l11)

	'''

	return llines
	

llines = sensitivity_other_missions()

#### plot PAHST sensitivity
#photometer
# l10 = plt.scatter(wave, F_source, label = 'PAHST spectrometer')
# llines.append(l10)

#spectrometer
l11 = plt.scatter(wave, F_source, label = 'PAHST spectrometer')
llines.append(l11)


plt.tight_layout()
plt.grid(alpha=0.4)

# add lines to explain what dashed and dotted are
dot = matplotlib.lines.Line2D([], [],
					   color='k',
					   ls='',
							  marker='o',
					   label='1 hr, $5\sigma$')
square = matplotlib.lines.Line2D([], [],
							   color='k',
							   ls='',
								 marker='s',
							   label='10 000s, 10$\sigma$')

leg1 = plt.legend(handles=[dot, square], loc=3, fontsize=9)
plt.legend(loc=2, fontsize=8, ncol=2).draggable()
plt.ylim(1e-23, 1e-16)
plt.gca().add_artist(leg1)

plt.yscale('log')
plt.xlabel('Wavelength [$\mu$m]')
plt.ylabel('Sensitivity [W/m$^2$]')
plt.xscale('log')

plt.savefig(f'PAHST_sensitivity_comparison.png', dpi = 300, bbox_inches = 'tight')
# plt.show()
plt.close()



'''
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
'''