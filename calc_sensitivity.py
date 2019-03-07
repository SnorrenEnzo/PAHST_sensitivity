import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from astropy import units as u
from astropy import constants as const

plt.rc('font', **{'size': 12})

matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["axes.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white"

def delta_lambda_spec(wave, R):
	"""
	Delta lambda for a given spectral resolution R of spectrometer.

	Parameters
	----------
	wave : Quantity
		Wavelength or astropy.units equivalent.
	R : float
		Spectral resolution at wave.

	Returns
	-------
	 : Quantity
		Delta lambda [micron]
	"""
	return wave.to(u.micron, equivalencies=u.spectral())/R


def angle_pix(wave, detector):
	"""
	Angle subtended by each pixel according to Table 1, Swinyard et al. (2004).

	Parameters
	----------
	wave : Quantity
		Wavelength or astropy.units equivalent.
	detector : string
		'imager' or 'spectrometer' to specify which detector is used.

	Returns
	-------
	angle_pix : Quantity
		Angle [arcsecond] subtended by a single pixel.
	"""
	angle_pix = np.zeros_like(wave.value)
	if detector == 'imager':
		angle_pix = 0.11*u.arcsecond
	elif detector == 'spectrometer':
		angle_pix = np.choose(
			wave.to(u.micron, equivalencies=u.spectral()) < 11*u.micron,
			[
				0.22,
				0.18
			]
		)*u.arcsecond
	else:
		raise ValueError("Unknown detector:"+str(detector))
	return angle_pix


def n_pix(D_tel, wave, detector):
	"""
	Number of pixels that the signal from a point source is spread over.

	Parameters
	----------
	D_tel : Quantity
		Diameter of telescope [L].
	wave : Quantity
		Wavelength or astropy.units equivalent.
	detector : string
		'imager' or 'spectrometer' to specify which detector is used.

	Returns
	-------
	 : float
		Number of pixels per point source.
	"""

	# Strehl ratio
	# Paper assumes a Strehl ratio of 82% at 5.6 micron, scaling as
	# 1 - C/lambda^2 where C is a constant
	C = (5.6*u.micron)**2*(1 - 0.82)
	Strehl_ratio = 1 - C/wave.to(u.micron, equivalencies=u.spectral())**2

	# angle subtended by diameter of first Airy ring, perfect optics
	angle_Airy_r = (
		2*wave.to(u.m, equivalencies=u.spectral())/D_tel)*u.rad

	# Angle subtended by diameter of first Airy ring, imperfect optics
	# (i.e. broaden the Airy ring in accordance with Strehl ratio)
	angle_Airy_r = angle_Airy_r/Strehl_ratio

	# Angular area of first Airy ring, imperfect optics
	area_Airy = np.pi*angle_Airy_r**2

	# Angular area subtended by each pixel
	area_pix = angle_pix(wave, detector)**2

	# Number of pixels that the signal from a point source is spread over
	return np.ceil((area_Airy/area_pix).decompose())


def n_pix_PAHST(wave):
	"""
	Number of pixels that the signal from a point source is spread over for
	PAHST.

	Parameters
	----------
	wave : Quantity
		Wavelength or astropy.units equivalent.

	Returns
	-------
	 : float
		Number of pixels per point source.
	"""
	n_pix = np.zeros_like(wave.value)
	n_pix = np.select(
		[
			(wave == 47*u.micron),
			(wave == 87*u.micron),
			(wave == 155*u.micron),
		],
		[
			# 4 pixels per beam, 12 beams per source
			4,
			# 4 pixels per beam, 12 beams per source
			4,
			# 2.75 pixels per beam, 12 beams per source
			2.75, #originally was 4
		]
	)
	return n_pix


def sigma_TF(spacecraft, S_back, t_frame):
	"""
	Random noise per pixel per frame.

	Parameters
	----------
	spacecraft : string
		'JWST' or 'PAHST' to specify spacecraft.
	S_back : Quantity
		Signal from background flux from the instrument, telescope, structure,
		and Zodiacal light [electrons/second]
	t_frame : Quantity
		Time per frame [T].

	Returns
	-------
	sigma_TF : float
		Random noise per pixel per frame [electrons].
	"""
	# Photoconductive gain
	G = 1

	# Gain dispersion
	beta = 1

	# In-band transmission
	# FIXME: give this unit of seconds, don't think it is the in band
	# transmission because then units in sigma_TF don't work out
	t_f = 0.5*u.second

	# use frame time instead
	t_f = t_frame

	# Dark current [electrons/s]
	I_dark = 0.03/u.second

	# Read noise per frame for the detector [electrons]
	if spacecraft == 'JWST':
		sigma_read = 20
	elif spacecraft == 'PAHST':
		sigma_read = 10
	else:
		raise ValueError('Unknown spacecraft:'+spacecraft)

	sigma_read = 20

	# Random noise per pixel per frame
	sigma_TF = np.sqrt(S_back*(G**2)*beta*t_f + I_dark*t_f + sigma_read**2)
	# sigma_TF = np.sqrt((S_back*(G**2)*beta*t_f)**2 + (I_dark*t_f)**2 + sigma_read**2)

	return sigma_TF


def sigma_T(spacecraft, t_i, t_frame, S_back):
	"""
	Total noise in integration time.

	Parameters
	----------
	spacecraft : string
		'JWST' or 'PAHST' to specify spacecraft.
	t_i : Quantity
		Integration time [T].
	t_frame : Quantity
		Time per frame [T].

	Returns
	-------
	sigma_T : float
		Total noise in integration time t_i [electrons].
	"""
	# Number of frame integrations
	n_f = np.floor((t_i/t_frame).decompose())

	# Total noise in integration time t_i
	sigma_T = np.sqrt(n_f)*sigma_TF(spacecraft, S_back, t_frame)

	return sigma_T


def t_read(spacecraft, wave, pattern_frame):
	"""
	Dead time associated with reading out a pixel (see Table 1).

	Parameters
	----------
	spacecraft : string
		'JWST' or 'PAHST' to specify spacecraft.
	wave : Quantity
		Wavelength or astropy.units equivalent.
	pattern_frame : ndarray
		Array of strings assumed to contain 'FAST' or 'SLOW'.

	Returns
	-------
	t_read : Quantity
		Read out dead time [s].
	"""
	if spacecraft == 'JWST':
		t_read = np.choose(
			pattern_frame == 'SLOW',
			[
				# where pattern_frame == 'FAST'
				np.choose(
					wave.to(u.micron, equivalencies=u.spectral()) \
					== 25.5*u.micron,
					[
						3,
						0.75
					]
				),
				# where pattern_frame == 'SLOW'
				30
			]
		)*u.second
	elif spacecraft == 'PAHST':
		t_read = np.ones_like(wave.value)*u.second
	else:
		raise ValueError('Unknown spacecraft:'+spacecraft)
	return t_read


def pixhit(spacecraft, t_frame):
	"""
	Fraction of pixels affected by cosmic ray hits.

	Parameters
	----------
	spacecraft : string
		'JWST' or 'PAHST' to specify spacecraft.
	t_frame : Quantity
		Time per frame [T].

	Returns
	-------
	 : float
		Fraction of pixels affected by cosmic ray hits.
	"""
	if spacecraft == 'JWST':
		# return 0.47*(t_frame/(1000*u.second)).decompose()
		return 1-(1-0.47)**(t_frame/(1000*u.second)).decompose() 
	elif spacecraft == 'PAHST':
		return 1-(1-0.47)**(t_frame/(1000*u.second)).decompose() #used to be 0.5
	else:
		raise ValueError('Unknown spacecraft:'+spacecraft)


def nonoppix(lifetime):
	"""
	Fraction of pixels that are non-operable.

	See Sec 4.1, para 1.

	Parameters
	----------
	lifetime : string
		'BOL' or 'EOL'

	Returns
	-------
	 : float
		Fraction of pixels that are non-operable.
	"""
	if lifetime == 'BOL':
		# At BOL, have 99% operable pixels
		return 0.01
	elif lifetime == 'EOL':
		# At EOL, have 95% operable pixels
		return 0.05
	else:
		raise ValueError("Unknown lifetime:"+lifetime)


def t_eff(spacecraft, t_i, t_frame, wave, pattern_frame, lifetime):
	"""
	Effective integration time.

	Parameters
	----------
	spacecraft : string
		'JWST' or 'PAHST' to specify spacecraft.
	t_i : Quantity
		Total integration time [T].
	t_frame : Quantity
		Time per frame [T].
	wave : Quantity
		Wavelength or astropy.units equivalent.
	pattern_frame : ndarray
		Array of strings assumed to contain 'FAST' or 'SLOW'.
	lifetime : string
		'BOL' or 'EOL'

	Returns
	-------
	 : Quantity
		Effective integration time [s].
	"""
	t_r = t_read(spacecraft, wave, pattern_frame)
	ph = pixhit(spacecraft, t_frame)
	nop = nonoppix(lifetime)

	return (t_i - t_r).to(u.second)*(1-ph-nop)


def eta_T_PAHST(wave, detector, lifetime):
	"""
	Transmission of the optics for PAHST.

	Parameters
	----------
	wave : Quantity
		Wavelength or astropy.units equivalent.

	Returns
	-------
	 : float or ndarray
		Transmission of the optics.
	"""
	#return 0.5*np.ones_like(wave.value)

	if detector == 'imager':
		eta_T = 0.45*np.ones_like(wave.value)
	elif detector == 'spectrometer':
		eta_T = 0.15*np.ones_like(wave.value)
	else:
		raise ValueError("Unknown detector:"+detector)

	if lifetime == 'BOL':
		return eta_T
	elif lifetime == 'EOL':
		return 0.8*eta_T
	else:
		raise ValueError("Unknown lifetime:"+lifetime)


def eta_T(wave, detector, lifetime):
	"""
	Transmission of the optics for JWST

	Parameters
	----------
	wave : Quantity
		Wavelength or astropy.units equivalent.
	detector : string
		'imager' or 'spectrometer' to specify which detector is used.
	lifetime : string
		'BOL' or 'EOL'

	Returns
	-------
	eta_T : float or ndarray
		Transmission of the optics.
	"""
	wave = wave.to(u.micron, equivalencies=u.spectral())
	eta_T = np.zeros_like(wave.value)
	if detector == 'imager':
		eta_T = np.select(
			[
				(5*u.micron <= wave) & (wave < 20*u.micron),
				(20*u.micron <= wave) & (wave <= 28*u.micron)
			],
			[
				0.5,
				0.4
			]
			)
	elif detector == 'spectrometer':
		eta_T = np.select(
			[
				(5*u.micron <= wave) & (wave < 18*u.micron),
				(18*u.micron <= wave) & (wave <= 28*u.micron),
			],
			[
				0.2,
				0.13
			]
		)
	else:
		raise ValueError("Unknown detector:"+detector)

	if lifetime == 'BOL':
		return eta_T
	elif lifetime == 'EOL':
		return 0.8*eta_T
	else:
		raise ValueError("Unknown lifetime:"+lifetime)


def eta_D(wave, detector):
	"""
	Quantum efficiency of the detectors.

	Parameters
	----------
	wave : Quantity
		Wavelength or astropy.units equivalent.
	detector : string
		'imager' or 'spectrometer' to specify which detector is used.

	Returns
	-------
	eta_D : float or ndarray
		Detector quantum efficiency.
	"""
	wave = wave.to(u.micron, equivalencies=u.spectral())
	eta_D = np.zeros_like(wave.value)
	if detector == 'imager':
		eta_D = np.select(
			[
				wave == 5.6*u.micron,
				(7.7*u.micron <= wave) & (wave < 12*u.micron),
				(12*u.micron <= wave) & (wave <= 25.5*u.micron)
			],
			[
				0.4,
				0.6,
				0.5
			]
			)
	elif detector == 'spectrometer':
		eta_D = np.select(
			[
				(wave == 6.4*u.micron) | (wave == 9.2*u.micron),
				(wave == 14.5*u.micron) | (wave == 22.5*u.micron),
			],
			[
				0.6,
				0.7
			]
		)
	else:
		raise ValueError("Unknown detector:"+detector)
	return eta_D


def eta_D_PAHST(wave):
	"""
	Quantum efficiency of the detectors of PAHST.

	Parameters
	----------
	wave : Quantity
		Wavelength or astropy.units equivalent.

	Returns
	-------
	 : float or ndarray
		Detector quantum efficiency.
	"""
	return 0.5*np.ones_like(wave.value)


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


def S_back_PAHST(wave, Delta_lambda, D_tel):
	"""
	Background signal for PAHST.

	Parameters
	----------
	wave : Quantity
		Wavelength or astropy.units equivalent.
	Delta_lambda : Quantity
		Channel width at each wavelength [L].

	Returns
	 : Quantity
		Background signal [electrons/s].
	"""
	# Assume that PAHST is only limited by Zodiacal light. Follow the
	# calculation in Swinyard et al. (2004)
	Z_F = 1.2
	B_5500 = planck_lam(wave, 5500*u.K)
	B_270 = planck_lam(wave, 270*u.K)
	# zodiacal light
	S_zod = Z_F*(3.5e-14*B_5500 + 3.58e-8*B_270)

	#telescope background, based on Swinyard et al. (2004)
	#Here, we have a 0.05K detector, 4K mirror and identical sun shield
	S_tel = 0.48*planck_lam(wave, 0.05*u.K) + \
			0.1*planck_lam(wave, 4*u.K) + \
			3e-5*planck_lam(wave, 92*u.K) + \
			9.9e-7*planck_lam(wave, 155*u.K)

	S_tot = S_zod + S_tel

	# pixel size
	D = 0.085*u.cm
	# half angle of the beam on to the detectors
	# theta = 0*u.deg
	theta = 1.22*wave.to(u.m)/D_tel/2*u.rad
	A_Omega = D**2*(2*np.pi*(1-np.cos((theta.to(u.rad)).value)))*u.sr
	eta_D = eta_D_PAHST(wave)

	return (S_tot*A_Omega*eta_D*wave*Delta_lambda/const.h/const.c).to(1/u.s)


def F_P(spacecraft, detector, wave, bandpass_or_R, S_back, pattern_frame, t_frame, lifetime, D_tel, n_sigma, A_tel, t_i):
	"""
	Minimum detectable spectral flux [photons/s/m**2/micron].

	Parameters
	----------
	spacecraft : string
		'JWST' or 'PAHST' to specify spacecraft.
	detector : string
		'imager' or 'spectrometer' to specify which detector is used.
	wave : Quantity
		Wavelength or astropy.units equivalent.
	bandpass_or_R : ndarray or Quantity
		For imager this is the bandpass [L], for spectrometer it is the
		spectral resolution R.
	S_back : Quantity
		Background signal [electrons/T].
	pattern_frame : ndarray
		Array of strings assumed to contain 'FAST' or 'SLOW'.
	t_frame : Quantity
		Time per frame [T].
	lifetime : string
		'BOL' or 'EOL'.
	D_tel : Quantity
		Telescope diameter [L**2].
	n_sigma : float
		Target SNR.
	A_tel : Quantity
		Effective collecting area of telescope over the defining wavelength
		band of the instrument.
	t_i : Quantity
		Total integration time [T].

	Returns
	F_P : Quantity
		Minimum detectable spectral flux [photons/s/m**2/micron].
	"""
	# need to degrade SNR at certain wavelengths (Sec. 3.5, para 2)
	n_sigma = n_sigma*np.ones_like(wave.value)
	if spacecraft == 'JWST':
		n_p = n_pix(D_tel, wave, detector)
		eT = eta_T(wave, detector, lifetime)
		eD = eta_D(wave, detector)
	elif spacecraft == 'PAHST':
		n_p = n_pix_PAHST(wave)
		eT = eta_T_PAHST(wave, detector, lifetime)
		eD = eta_D_PAHST(wave)
	else:
		raise ValueError('Unknown spacecraft:'+spacecraft)
	sigT = sigma_T(spacecraft, t_i, t_frame, S_back)
	teff = t_eff(spacecraft, t_i, t_frame, wave, pattern_frame, lifetime)
	if detector == 'imager':
		Delta_lambda = bandpass_or_R
		if spacecraft == 'JWST':
			# Extraction penalty factors are introduced at 5.6 micron and for
			# the spectrometer operation at 6.4 and 9.2 micron ... degrade the
			# signal to noise ratio by 10% (see Sec 3.5, para 2)
			n_sigma[
				wave.to(u.micron, equivalencies=u.spectral())
				== 5.6*u.micron] *= (1+1./9)
	elif detector == 'spectrometer':
		Delta_lambda = delta_lambda_spec(wave, bandpass_or_R)
		if spacecraft == 'JWST':
			# Extraction penalty factors are introduced at 5.6 micron and for
			# the spectrometer operation at 6.4 and 9.2 micron ... degrade the
			# signal to noise ratio by 10% (see Sec 3.5, para 2)
			n_sigma[
				wave.to(u.micron, equivalencies=u.spectral())
				== 6.4*u.micron] *= (1+1./9)
			n_sigma[
				wave.to(u.micron, equivalencies=u.spectral())
				== 9.2*u.micron] *= (1+1./9)
	else:
		raise ValueError("Unknown detector:"+detector)

	F_P = (
		(np.sqrt(n_p)*n_sigma*sigT) / (A_tel*eT*eD*teff*Delta_lambda)
	).to(1/u.second/u.m**2/u.micron)
	return F_P

def calc_JWST_sensitivity():
	###############################################################################
	# JWST instrument characteristics etc.

	# telescope diameter
	D_tel = 6*u.m

	# Effective collecting area of telescope over the defining wavelength
	# band of the instrument.
	A_tel = 25*u.m**2

	# target SNR
	SNR = 10
	# target integration time
	t_i = 10000*u.s
	###############################################################################

	###############################################################################
	# JWST imager

	# These are the input values in the paper
	Band = np.array(
	    [5.6, 7.7, 10, 11.3, 12.8, 15, 18, 21, 25.5])*u.micron
	Bandpass = np.array(
	    [1.2, 2.2, 2, 0.8, 2.5, 4, 3, 5, 3.9])*u.micron
	EstimatedBackground = np.array(
	    [6, 45, 94, 52, 222, 526, 672, 2354, 7677])/u.second
	FramePattern = np.array([
	    'SLOW', 'FAST', 'FAST', 'FAST', 'FAST', 'FAST', 'FAST', 'FAST', 'FAST'])
	FrameDuration = np.array([480, 120, 120, 120, 120, 60, 30, 15, 6])*u.second

	# These are the results reported in the paper
	DetectionLimitEOL = np.array(
	    [0.19, 0.28, 0.7, 1.7, 1.4, 1.8, 4.3, 7.3, 29])*u.microJansky
	DetectionLimitBOL = np.array(
	    [0.15, 0.23, 0.5, 1.15, 0.9, 1.1, 3.1, 5.7, 25])*u.microJansky

	# EOL F_lambda^P
	f_p_l_eol = F_P('JWST', 'imager', Band, Bandpass,
	                EstimatedBackground, FramePattern,
	                FrameDuration, 'EOL', D_tel, SNR, A_tel, t_i)
	# convert to F_nu^P
	f_p_nu_eol = Band/Band.to(u.Hz, equivalencies=u.spectral()) * f_p_l_eol
	# convert to F_nu
	f_nu_eol = (
	    f_p_nu_eol*Band.to(u.eV, equivalencies=u.spectral())).to(u.microJansky)

	# BOL F_lambda^P
	f_p_l_bol = F_P('JWST', 'imager', Band, Bandpass,
	                EstimatedBackground, FramePattern,
	                FrameDuration, 'BOL', D_tel, SNR, A_tel, t_i)
	# convert to F_nu^P
	f_p_nu_bol = Band/Band.to(u.Hz, equivalencies=u.spectral()) * f_p_l_bol
	# convert to F_nu
	f_nu_bol = (
	    f_p_nu_bol*Band.to(u.eV, equivalencies=u.spectral())).to(u.microJansky)

	fig, ax = plt.subplots()
	ax.set_xlabel('table column')
	ax.set_ylabel(r'Detection limit$\,[\mu$Jy]')
	ax.set_yscale('log')
	ax.set_title('JWST Imager')
	ax.plot(DetectionLimitBOL, label='Paper, BOL')
	ax.plot(DetectionLimitEOL, label='Paper, EOL')
	ax.plot(f_nu_bol, label='Calculated, BOL')
	ax.plot(f_nu_eol, label='Calculated, EOL')
	ax.legend()
	plt.show()
	###############################################################################

	###############################################################################
	# JWST spectrometer

	# These are the input values in the paper
	Wavelength = np.array(
	    [6.4, 9.2, 14.5, 22.5])*u.micron
	R = np.array([2400, 2400, 1600, 1200])
	EstimatedBackground = np.array([0.04, 0.08, 0.5, 3.5])/u.second
	FramePattern = np.array(['SLOW', 'SLOW', 'SLOW', 'SLOW'])
	FrameDuration = np.array([960, 960, 480, 480])*u.second

	# These are the results reported in the paper
	DetectionLimitEOL = np.array([1.2, 1, 1.2, 5.6])*1e-20*u.W/u.m**2
	DetectionLimitBOL = np.array([0.8, 0.75, 0.8, 5])*1e-20*u.W/u.m**2

	# EOL F_lambda^P
	f_p_l_eol = F_P('JWST', 'spectrometer', Wavelength, R, EstimatedBackground,
	                FramePattern, FrameDuration, 'EOL', D_tel, SNR, A_tel, t_i)
	# convert to F_lambda
	f_l_eol = Wavelength.to(u.eV, equivalencies=u.spectral()) * f_p_l_eol
	# convert to F
	f_eol = f_l_eol * Wavelength / R

	# BOL F_lambda^P
	f_p_l_bol = F_P('JWST', 'spectrometer', Wavelength, R, EstimatedBackground,
	                FramePattern, FrameDuration, 'BOL', D_tel, SNR, A_tel, t_i)
	# convert to F_lambda
	f_l_bol = Wavelength.to(u.eV, equivalencies=u.spectral()) * f_p_l_bol
	# convert to F
	f_bol = f_l_bol * Wavelength / R

	fig, ax = plt.subplots()
	ax.set_xlabel('table column')
	ax.set_ylabel(r'Detection limit$\,$[W/m$^2$]')
	ax.set_yscale('log')
	ax.set_title('JWST Spectrometer')
	ax.plot(DetectionLimitBOL, label='Paper, BOL')
	ax.plot(DetectionLimitEOL, label='Paper, EOL')
	ax.plot((f_bol).to(u.W/u.m**2),
	        label='Calculated, BOL')
	ax.plot((f_eol).to(u.W/u.m**2),
	        label='Calculated, EOL')
	ax.legend()
	plt.show()
	###############################################################################

def calc_PAHST_sensitivity():
	############################################################################
	# PAHST instrument characteristics etc.

	# telescope diameter
	D_tel = 8*u.m

	# Effective collecting area of telescope over the defining wavelength
	# band of the instrument.
	A_tel = np.pi*(D_tel/2)**2

	#target sigma
	target_sigma = 5

	# target SNR
	SNR = target_sigma**2 #10
	# target integration time
	t_i = 3600*u.s #10000*u.s
	############################################################################

	############################################################################
	# PAHST imager
	print('Imager')

	Band = np.array([47, 87, 155])*u.micron
	Bandpass = np.array([34, 46, 90])*u.micron
	# calculate background
	EstimatedBackground = S_back_PAHST(Band, Bandpass, D_tel)
	FramePattern = np.array([None, None, None])
	# FrameDuration = np.array([3600, 3600, 3600])*u.second
	FrameDuration = np.array([1000, 1000, 1000])*u.second

	# EOL F_lambda^P
	f_p_l_eol = F_P('PAHST', 'imager', Band, Bandpass,
					EstimatedBackground, FramePattern,
					FrameDuration, 'EOL', D_tel, SNR, A_tel, t_i)
	# convert to F_nu^P
	f_p_nu_eol = Band/Band.to(u.Hz, equivalencies=u.spectral()) * f_p_l_eol
	# convert to F_nu
	f_nu_eol = (
		f_p_nu_eol*Band.to(u.eV, equivalencies=u.spectral())).to(u.microJansky)
	print('EOL', f_nu_eol)

	# BOL F_lambda^P
	f_p_l_bol = F_P('PAHST', 'imager', Band, Bandpass,
					EstimatedBackground, FramePattern,
					FrameDuration, 'BOL', D_tel, SNR, A_tel, t_i)
	# convert to F_nu^P
	f_p_nu_bol = Band/Band.to(u.Hz, equivalencies=u.spectral()) * f_p_l_bol
	# convert to F_nu
	f_nu_bol = (
		f_p_nu_bol*Band.to(u.eV, equivalencies=u.spectral())).to(u.microJansky)
	print('BOL', f_nu_bol)

	# "save" for later
	f_PAHST_imager = ((f_nu_bol*Band.to(u.Hz, equivalencies=u.spectral()) * \
	  Bandpass/Band).to(u.W/u.m**2)).value[:]*u.W/u.m**2

	############################################################################


	############################################################################
	# PAHST spectrometer
	print('Spectrometer')

	# These are the input values in the paper
	Wavelength = np.array([47, 87, 155])*u.micron
	R = np.array([2100, 2250, 1250])
	# calculate background
	EstimatedBackground = S_back_PAHST(
		Wavelength, delta_lambda_spec(Wavelength, R), D_tel)


	FramePattern = np.array([None, None, None])
	# FrameDuration = np.array([3600, 3600, 3600])*u.second
	FrameDuration = np.array([1000, 1000, 1000])*u.second

	# EOL F_lambda^P
	f_p_l_eol = F_P('PAHST', 'spectrometer', Wavelength, R, EstimatedBackground,
					FramePattern, FrameDuration, 'EOL', D_tel, SNR, A_tel, t_i)
	# convert to F_lambda
	f_l_eol = Wavelength.to(u.eV, equivalencies=u.spectral()) * f_p_l_eol
	# convert to F
	f_eol = f_l_eol * Wavelength / R
	print('EOL', f_eol.to(u.W/u.m**2))

	# BOL F_lambda^P
	f_p_l_bol = F_P('PAHST', 'spectrometer', Wavelength, R, EstimatedBackground,
					FramePattern, FrameDuration, 'BOL', D_tel, SNR, A_tel, t_i)
	# convert to F_lambda
	f_l_bol = Wavelength.to(u.eV, equivalencies=u.spectral()) * f_p_l_bol
	# convert to F
	f_bol = f_l_bol * Wavelength / R
	# "save" for later
	f_PAHST_spectrometer = f_bol.to(u.W/u.m**2)

	print('BOL', f_bol.to(u.W/u.m**2))
	############################################################################


	def createProxyLine(line, label):
		"""																																																									   
		Create a proxy line with the properties of line and a label but with no																																								   
		data. If added to the lines of an axe object, it will therefore not be																																									
		plotted, but, since it has a label, it will show up in the legend.																																										
		"""
		l = copy.copy(line)
		l.set_xdata([])
		l.set_ydata([])
		l.set_label(label)
		l.figure = None
		l.axes = None
		return l


	"""
	Plot the sensitivity of other instruments out there
	"""

	# transparency
	aa = 0.8

	###
	n_inst = 12
	colors = plt.cm.viridis(np.linspace(0,1,n_inst))

	plt.figure(figsize=(8,5))
	plt.yscale('log')
	plt.xlabel('Wavelength [$\mu$m]')
	plt.ylabel('Sensitivity [W/m$^2$]')
	plt.xscale('log')

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
	miri_ll = [5.6, 7.7, 10, 11.3, 12.8, 15, 18, 21, 25.5] *u.micrometer
	miri_nu = const.c / miri_ll.to(u.meter)
	miri_sy = [0.16, 0.25, 0.56, 1.35, 0.84, 1.39, 3.46, 7.09, 26.2] * u.microjansky	# 10,000 s, 10 sigma 
	miri_sy_wm2 = (miri_sy * miri_nu).to(u.W/u.m**2)
	l3, = plt.plot(miri_ll, miri_sy_wm2, ls='', marker='s', \
		alpha=aa, label='JWST, MIRI', \
		color=colors[3])

	llines.append(l3)
	#
	# NIRCam: https://jwst-docs.stsci.edu/display/JTI/NIRCam+Filters Tables 2 and 3
	ncam_ll_w = [0.7, 0.90, 1.15, 1.50, 2.00, 2.77, 3.22, 3.56, 4.44] *u.micrometer
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

	# PACS: https://www.cosmos.esa.int/documents/12133/996891/PACS+Spectrometer+Fact+Sheet
	pacs_ll =  [70, 92, 140, 180] *u.micrometer
	pacs_sy =  [7.2e-18, 6.8e-18, 2.4e-18, 3.6e-18] *u.W/u.m**2
	l9, = plt.plot(pacs_ll, pacs_sy, ls='', marker='o', \
		alpha=aa, color = colors[9], label='Herschel, PACS spectr.')
	llines.append(l9)

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
	#plt.ylim(1e-23, 1e-22)
	plt.gca().add_artist(leg1)

	iteration = 10
	# plt.title(f'Iteration: {iteration}')
	# plt.savefig(f'{iteration}_sensitivity_plot.png', dpi = 300, bbox_inches = 'tight')
	plt.savefig(f'PAHST_sensitivity_plot.png', dpi = 300, bbox_inches = 'tight')
	plt.show()

calc_PAHST_sensitivity()
# calc_JWST_sensitivity()
