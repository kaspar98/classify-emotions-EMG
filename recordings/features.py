import numpy as np
import scipy.signal as ssignal
import scipy


class SignalFeatures:
	"""
	Signal Features class computes 16-features for an EMG signal,
	recommended in xxx

	...

	Parameters:
		signal (list or array): 1D TimeSeries Filtered signal

	Attributes:


	"""

	def __init__(self, signal=None):
		if signal is None:
			raise Exception("Signal required for features to be extracted")
		else:
			self.signal = np.array(signal)
			self.N = len(self.signal)

	def mav(self):
		""" Computes mean absolute value (mav) of signal

		Returns
		-------
		float
			representing mav of signal
		"""
		return np.mean(abs(self.signal))

	def mavfd(self):
		""" Computes mean absolute value of first difference (mavfd) from signal
			Sums absolute value of subsequent difference in signal amplitude

		Returns
		------
		float
			representing mavfd of signal
		"""

		return np.mean(abs(np.diff(self.signal, n=1)))

	def mavsd(self):
		""" Computes mean absolute value of second difference (mavsd) from signal
			Sums absolute value of subsequent differences (interval = 2) in signal amplitude

		Returns
		------
		float
			representing mavsd of signal
		"""

		return np.mean(abs(np.diff(self.signal, n=2)))

	def peak(self):
		""" Obtains maximum amplitude recorded for signal

		Returns
		------
		float
			peak
		"""
		return max(self.signal)

	def rms(self):
		""" Computes Root Mean Square (rms) of signal

		Returns
		-------
		float
			root mean square value
		"""
		return np.sqrt(np.mean(np.square(self.signal)))

	def zc(self):
		""" Computes Nr. of Zero Crossings (zc): number of time signal changes from positive to negative

		Parameters
		---------
		threshold : float

		Returns
		-------
		zc: int
			number of zero-crossings observed, given threshold
		"""

		zcs = np.where(np.diff(np.signbit(self.signal)))[0]
		nr_zcs = len(zcs)

		return nr_zcs

	def fmed(self):
		""" Computes frequency that divides the spectrum into two regions with equal power

		Returns
		------
		fmed: float
			median frequency
		"""

		f, P = ssignal.welch(self.signal, window='hanning', noverlap=0, nfft=int(256.))

		area_freq = scipy.integrate.cumtrapz(P, f, initial=0)
		total_power = area_freq[-1]
		median_freq = f[np.where(area_freq >= total_power / 2)[0][0]]

		return median_freq

	def fmode(self):
		""" Computes maximum value of FFT-Transformed signal
			Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4712064/

		Returns
		------
		fmode: complex number
		"""
		return max(np.fft.fft(self.signal))

	def fmean(self):
		""" Computes ...
			Reference: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180526

		Returns
		------
		mnf: float
			Mean Frequency
		"""
		f, P = ssignal.welch(self.signal, window='hanning', noverlap=0, nfft=int(256.))
		xi_ = [(f[i] * P[i]) for i in range(len(f))]
		xj_ = np.mean([P[i] for i in range(len(P))])
		mnf = np.mean(xi_/xj_)

		return mnf

	# def cf(self):
		""" Computes ... Central Frequency
			Reference: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180526

		Returns
		------
		mnf: float
			Mean Frequency
		"""
		# return

	def sampen(self, m=2, r=None):
		""" Computes sample entropy of time-series data
			Reference: https://www.mathworks.com/matlabcentral/fileexchange/35784-sample-entropy
			Reference 2: https://en.wikipedia.org/wiki/Sample_entropy


		Parameters
		---------
		m: int
			Embedding Dimension of signal array
		r: int
			Tolerance: % of deviation

		Returns
		------
		mnf: float
			Mean Frequency
		"""

		r = 0.2 * self.std() if r is None else r

		# Split time series and save all templates of length m
		xmi = np.array([self.signal[i: i + m] for i in range(self.N - m)])
		xmj = np.array([self.signal[i: i + m] for i in range(self.N - m + 1)])

		# Save all matches minus the self-match, compute B
		B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

		# Similar for computing A
		m += 1
		xm = np.array([self.signal[i: i + m] for i in range(self.N - m + 1)])

		A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

		# Return SampEn
		return -np.log(A / B)

	def apen(self, m=2, r=None):
		""" Computes approximate entropy of time-series data
			Code Reference: https://gist.github.com/DustinAlandzes/a835909ffd15b9927820d175a48dee41

		Parameters
		---------
		m: int
			Embedding Dimension  of signal array

		r: int
			Tolerance

		Returns
		------
		float
			ApEn
		"""

		r = 0.2 * self.std() if r is None else r

		def _maxdist(x_i, x_j):
			return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

		def _phi(m):
			x = [[self.signal[j] for j in range(i, i + m - 1 + 1)] for i in range(self.N - m + 1)]
			C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (self.N - m + 1.0) for x_i in x]
			return (self.N - m + 1.0) ** (-1) * sum(np.log(C))

		return abs(_phi(m + 1) - _phi(m))

	def var(self):
		""" Computes variance of signal values

		Returns
		------
		float
			signal var
		"""
		return np.var(self.signal)

	def std(self):
		""" Computes standard deviation of signal values

		Returns
		------
		float
			signal std
		"""
		return np.std(self.signal)

	def srange(self):
		""" Computes range of maximum and minimum amplitude

		Returns
		------
		float
			signal srange
		"""
		return max(self.signal) - min(self.signal)

	def intrange(self):
		""" Computes inter-quartile range of signal values

		Returns
		------
		float
			signal iqr
		"""
		return scipy.stats.iqr(self.signal)

	def __features__(self):
		""" Get names of all features (methods) implemented

		Returns
		-------
		List
			List of feature names
		"""

		feature_list_ = [method for method in dir(SignalFeatures) if method.startswith('__') is False]

		return feature_list_

	def __cfeatures__(self):
		""" Computes all 16 features

		Returns
		-------
		Dictionary
			Key-Value containing feature names and values
		"""

		feature_dict_ = {}

		for feature in self.__features__():
			feature_dict_[feature] = eval(f'self.{feature}()')

		return feature_dict_


