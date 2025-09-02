from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.fftpack
from src.shared.numpy.arrays import get_boundaries, zero_padding

class POCFunction:
    """
    Phase-Only Correlation (POC) function for image registration.

    This class implements the POC method for registering two images by calculating the phase correlation.

    :param ref_image: The reference image as a numpy array.
    :type ref_image: np.ndarray
    :param cmp_image: The comparison image as a numpy array.
    :type cmp_image: np.ndarray
    """
    def __init__(self, ref_image: np.ndarray, cmp_image: np.ndarray):
        assert ref_image.shape == cmp_image.shape, f"The two vectors must be the same shape"
        assert len(ref_image.shape) == 2, f"The two vectors must be 2D"
        self.ref_image = ref_image
        self.cmp_image = cmp_image

    def evaluate(self) -> np.ndarray:
        """
        Evaluate the POC function.

        This method applies windowing, calculates the FFT, cross-phase spectrum, spectral weighting,
        and finally the POC to register the images.

        :return: The POC result as a numpy array.
        :rtype: np.ndarray
        """
        self._apply_windowing()
        F, G = self._calculate_fft()
        R = self._calculate_cross_phase_spectrum(F, G)
        if R is None:
            return np.zeros(self.ref_image.shape)
        R = self._apply_spectral_weighting(R)
        poc = self._calculate_poc(R)
        return poc

    def _apply_windowing(self):
        """
        Apply Hanning windowing to the images.

        This method applies a 2D Hanning window to both the reference and comparison images.
        """
        hanning_window_x = np.hanning(self.ref_image.shape[0])
        hanning_window_y = np.hanning(self.ref_image.shape[1])
        hanning_window_2d = hanning_window_x.reshape(
            hanning_window_x.shape[0], 1) * hanning_window_y
        self.ref_image, self.cmp_image = [self.ref_image, self.cmp_image] * hanning_window_2d

    def _calculate_fft(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the FFT of the images.

        This method calculates the 2D FFT of both the reference and comparison images.

        :return: A tuple containing the FFT of the reference and comparison images.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        F = scipy.fftpack.fft2(self.ref_image)
        G = scipy.fftpack.fft2(self.cmp_image)
        return F, G

    @staticmethod
    def _calculate_cross_phase_spectrum(F: np.ndarray, G: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate the cross-phase spectrum.

        This method calculates the cross-phase spectrum of the FFTs of the reference and comparison images.

        :param F: The FFT of the reference image.
        :type F: np.ndarray
        :param G: The FFT of the comparison image.
        :type G: np.ndarray
        :return: The cross-phase spectrum as a numpy array, or None if the spectrum is zero.
        :rtype: Optional[np.ndarray]
        """
        G_ = np.conj(G)
        if np.all(F * G_ == 0):
            return None
        R = F * G_ / np.abs(F * G_)
        return scipy.fftpack.fftshift(R)

    def _apply_spectral_weighting(self, R: np.ndarray) -> np.ndarray:
        """
        Apply spectral weighting to the cross-phase spectrum.

        This method applies a low-pass filter to the cross-phase spectrum.

        :param R: The cross-phase spectrum.
        :type R: np.ndarray
        :return: The weighted cross-phase spectrum.
        :rtype: np.ndarray
        """
        M = np.floor([self.ref_image.shape[0] / 2.0, self.ref_image.shape[1] / 2.0])
        U = M / 2.0  # U = [U1, U2]
        low_pass_filter = np.ones([int(M[0]) + 1, int(M[1]) + 1])
        low_pass_filter = zero_padding(low_pass_filter, self.ref_image.shape, U)
        R = R * low_pass_filter
        return scipy.fftpack.fftshift(R)

    def _calculate_poc(self, R: np.ndarray) -> np.ndarray:
        """
        Calculate the Phase-Only Correlation (POC).

        This method calculates the POC by performing the inverse FFT of the weighted cross-phase spectrum.

        :param R: The weighted cross-phase spectrum.
        :type R: np.ndarray
        :return: The POC result as a numpy array.
        :rtype: np.ndarray
        """
        return scipy.fftpack.fftshift(np.real(scipy.fftpack.ifft2(R)))

class POCModel:
    """
    Phase-Only Correlation (POC) model for image registration.

    This class implements the POC model for evaluating the phase correlation between images.

    :param al: The alpha parameter for the POC model.
    :type al: float
    :param dt1: The delta t1 parameter for the POC model.
    :type dt1: float
    :param dt2: The delta t2 parameter for the POC model.
    :type dt2: float
    :param poc: The POC array.
    :type poc: np.ndarray
    :param fitting_area: The fitting area array.
    :type fitting_area: np.ndarray
    """
    def __init__(self, al: float, dt1: float, dt2: float, poc: np.ndarray, fitting_area: np.ndarray):
        assert len(poc.shape) == 2, f"the POC array should have a 2d shape"
        assert fitting_area.shape[0] == 2, f"The fitting area should be a 2d array"
        self.al = al
        self.dt1 = dt1
        self.dt2 = dt2
        self.poc = poc
        self.fitting_area = fitting_area

    def evaluate(self, n1: float, n2: float) -> np.ndarray:
        """
        Evaluate the POC model.

        This method calculates the POC value for the given n1 and n2 parameters.

        :param n1: The n1 parameter.
        :type n1: float
        :param n2: The n2 parameter.
        :type n2: float
        :return: The POC value as a numpy array.
        :rtype: np.ndarray
        """
        N1, N2 = self.poc.shape
        V1, V2 = self._calculate_v_values()
        return self._calculate_poc_value(n1, n2, N1, N2, V1, V2)

    def _calculate_v_values(self) -> List[np.ndarray]:
        """
        Calculate the V values for the POC model.

        This method calculates the V1 and V2 values based on the fitting area.

        :return: A list containing the V1 and V2 values.
        :rtype: List[np.ndarray]
        """
        return map(lambda x: 2 * x + 1, self.fitting_area)

    def _calculate_poc_value(self, n1: float, n2: float, N1: float, N2: float, V1: float, V2: float) -> float:
        """
        Calculate the POC value.

        This method calculates the POC value based on the given parameters.

        :param n1: The n1 parameter.
        :type n1: float
        :param n2: The n2 parameter.
        :type n2: float
        :param N1: The N1 parameter.
        :type N1: float
        :param N2: The N2 parameter.
        :type N2: float
        :param V1: The V1 parameter.
        :type V1: float
        :param V2: The V2 parameter.
        :type V2: float
        :return: The calculated POC value.
        :rtype: float
        """
        return self.al * np.sin((n1 + self.dt1) * V1 / N1 * np.pi) * np.sin((n2 + self.dt2) * V2 / N2 * np.pi) \
            / (np.sin((n1 + self.dt1) * np.pi / N1) * np.sin((n2 + self.dt2) * np.pi / N2) * (N1 * N2))
