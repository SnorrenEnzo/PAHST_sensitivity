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

wave = np.linspace(20, 200, 1000)*u.micron
# wave = np.linspace(5, 20, 1000)*u.micron

# Zodiacal light calculation in Swinyard et al. (2004)
Z_F = 1.2
B_5500 = planck_lam(wave, 5500*u.K)
B_270 = planck_lam(wave, 270*u.K)
# zodiacal light spectral radiance (per steradian)
S_zod = Z_F*(3.5e-14*B_5500 + 3.58e-8*B_270)

telescope = 'PAHST'

#### Calculate the spectral radiance of the telescope background
# based on the model from Swinyard et al. (2004)
if telescope == 'JWST':
	#exact copy from Swinyard et al. (2004)
	S_tel = 0.48*planck_lam(wave, 38.3*u.K) + \
			0.1*planck_lam(wave, 44.5*u.K) + \
			3e-5*planck_lam(wave, 92*u.K) + \
			9.9e-7*planck_lam(wave, 155*u.K)
	S_shield_92 = 3e-5*planck_lam(wave, 92*u.K)
	S_shield_155 = 9.9e-7*planck_lam(wave, 155*u.K)
elif telescope == 'PAHST':
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
	S_tel = S_dect_mirror + S_sunshield
elif telescope == 'SPICA':
	S_tel = 0.48*planck_lam(wave, 0.05*u.K) + \
			0.1*planck_lam(wave, 8*u.K) + \
			3e-5*5*planck_lam(wave, 92*u.K) + \
			9.9e-7*5*planck_lam(wave, 155*u.K)
		#detector/instrument
		#main mirror temperature
		#sun shield
		#sun shield

S_tot = S_zod + S_tel


if telescope == 'PAHST':
	#### some semi-fixed values
	#integration time
	t_i = 20*u.h
	#desired signal to noise
	# desired_SN = 10
	#FWHM of telescope; galaxy is effectively a point source
	fwhm = 1.8 * u.arcsecond
	#spectral resolution at 55 microns from Fig 13 of PAHST paper, page 37
	R_55 = 2666.
	#telescope diameter
	d_tel = 8*u.m

	##### spectral irradiance of the galaxy, source: fig 3
	S_gal = 4.5e-6*u.Jy
	# also calculate the wavelength of this emission
	wave_rest_frame = 7.7*u.micron #PAH line
	#redshift
	z = 6.
	wave_redshifted = wave_rest_frame * (z+1)
	print(f'7.7 micron PAH feature redshifted to {wave_redshifted}')
	plt.scatter(wave_redshifted, S_gal, marker = '*', color = 'r', label = 'Galaxy emission')


	#### Here we determine the background spectral irradiance in an area of the sky
	#convert this FWHM to a solid angle
	gal_solid_angle = ((fwhm/2.)**2 * np.pi).to(u.sr)

	#conversion factor from per wavelength to per frequency and taking into
	#account the angular size of the diffraction pattern of a galaxy
	conv = wave.to(u.m)**2/const.c * gal_solid_angle
	#also convert units
	conv_units = u.Jy

	#### calculate signal to noise
	#first find location of closest wavelength in the wave array
	loc = np.argmin(np.abs(wave - wave_redshifted))

	#calculate SN
	SN = S_gal / (S_tot[loc]*conv[loc]).to(u.Jy) * (t_i.to(u.s)/(1.*u.s))**0.5
	print(f'\nBackground model S/N: {round(SN.value, 6)}')
	#desired S/N: 10
	#https://www.eso.org/~ohainaut/ccd/sn.html

	#calculate number of sources observed in 9 years
	print(f'Number of sources in 9 years: {1000*20/t_i.value}')

	#calculate ratio between total background radiance and zodiacal radiance
	#at the wavelength of the galaxy
	# spec_ratio = S_tot[loc]/S_zod[loc]
	# print(f'Spectral ratio: {spec_ratio}')


	#### calculate S/N using only the NEP
	#more info: https://en.wikipedia.org/wiki/Noise-equivalent_power
	nep = 5e-20*u.W/u.Hz**0.5  #Kenyon et al. 2006, page 38 of PAHST paper
	#with this NEP one can detect a signal of 5e-20 W with S/N = 1 after
	#0.5s of integration time

	#now determine the wavelength bin at 7.7 microns redshifted to 54 
	#microns
	delta_lambda = wave_redshifted/R_55
	#convert to frequency bin
	delta_f = (const.c/(wave_redshifted.to(u.m)**2)*delta_lambda.to(u.m)).to(u.Hz)

	#convert spectral irradiance irradiance of galaxy to flux
	F_gal = S_gal.to(u.W/u.m**2/u.Hz) * delta_f

	#calculate telescope area
	A_tel = np.pi*(d_tel/2)**2

	#calculate radiant flux of galaxy
	P_gal = F_gal * A_tel

	#determine SN using only the NEP by dividing it by the NEP
	SN_nep_based = (P_gal/(nep))*1*(t_i.to((u.Hz)**-1))**0.5
	print(f'\nOnly using NEP: S/N = {SN_nep_based}')
else:
	conv = wave.to(u.m)**2/const.c
	#also convert units
	conv_units = u.Jy/u.sr

#### plot spectral radiance (per solid angle)
#main emission elements
plt.plot(wave, (S_tel*conv).to(conv_units), label = r'$S_{tel}$', color = '#1B26C7')
plt.plot(wave, (S_zod*conv).to(conv_units), label = r'$S_{Zod}$', color = '#E82C0C')
plt.plot(wave, (S_tot*conv).to(conv_units), label = r'$S_{Tot}$', color = '#F7D203', linestyle = '--')

#elements of the telescope
plt.plot(wave, (S_shield_1*conv).to(conv_units), label = f'S_{T_sunshield_1}', color = '#5084FF')
plt.plot(wave, (S_shield_2*conv).to(conv_units), label = f'S_{T_sunshield_2}', color = '#3BE9FF')

# S_shield_92 = 3e-5*planck_lam(wave, 92.*u.K)
# plt.plot(wave, (S_shield_92*conv).to(conv_units), label = r'$S_{92K}$', color = 'black')

plt.grid(alpha=0.4)
plt.yscale('log')
plt.ylabel('Spectral radiance [Jy/sr]')
plt.xlabel(r'Wavelength $[\mu m]$')
plt.title(f'{telescope} spectral radiance of telescope and zodiacal background')

# plt.ylim((1e-6, 1e0))

plt.legend(loc = 'best')
plt.savefig(f'{telescope}_spectral_radiance_background.png', dpi = 300, bbox_inches = 'tight')
plt.show()