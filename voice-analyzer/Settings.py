from utils import get_integer
from scipy.signal import firwin
from Definitions import Communicates, Filters


class FilterSettings:
    def __init__(self, chunk, rate):
        self.chunk = chunk
        self.rate = rate

        self.filter_type_enum = 0
        self.fir_filter = None
        self.f_min = 0
        self.f_max = 0

    def set_filter_frequencies(self, f_min_text, f_max_text):

        f_min = get_integer(f_min_text)

        if f_min is not None:
            if 0 <= f_min < 22000:
                self.f_min = f_min
            else:
                return Communicates.FREQUENCY_RANGE_ERROR_MESSAGE

        f_max = get_integer(f_max_text)

        if f_max is not None:
            if f_max > self.f_min and (0 < f_max <= 22000):
                self.f_max = f_max
            else:
                return Communicates.FREQUENCY_RANGE_ERROR_MESSAGE

        return

    def create_filter(self):
        if self.filter_type_enum in (Filters.LOW_PASS, Filters.HIGH_PASS):
            if self.f_min is None:
                return Communicates.CUTOFF_FREQUENCY_ERROR_MESSAGE
            else:
                pass_zero = "lowpass" if self.filter_type_enum == Filters.LOW_PASS else "highpass"
                self.fir_filter = firwin(numtaps=self.chunk,
                                         cutoff=self.f_min,
                                         fs=self.rate,
                                         pass_zero=pass_zero)
                return

        if self.filter_type_enum in (Filters.BAND_PASS, Filters.BAND_STOP):
            if self.f_min is None or self.f_max is None:
                return Communicates.CUTOFF_FREQUENCY_ERROR_MESSAGE
            else:
                pass_zero = "bandpass" if self.filter_type_enum == Filters.BAND_PASS else "bandstop"
                self.fir_filter = firwin(numtaps=self.chunk,
                                         cutoff=[self.f_min, self.f_max],
                                         fs=self.rate,
                                         pass_zero=pass_zero)
                return
        return


class Settings:
    def __init__(self):
        self.RATE = 44000
        self.CHUNK = 1023
        self.filters = FilterSettings(self.CHUNK, self.RATE)

        # Oscillogram and spectrogram display settings
        self.volume = 50

        self.oscillogram_t_min = 0
        self.oscillogram_t_max = 1000

        self.spectrogram_f_min = 0
        self.spectrogram_f_max = 5000
        self.spectrogram_chunks_per_ftt = 5    # number of chunks used to calculate Fourier transform

        self.modulation = False
        self.modulation_freq_shift = 0

        # Action settings

        self.record = False
        self.preview = False
        self.stop = False
        self.play = False
        self.listen = False

    def set_oscillogram_params(self, t_min_text, t_max_text):

        t_min = get_integer(t_min_text)

        if t_min is not None:
            if 0 <= t_min < 44000:
                self.oscillogram_t_min = t_min
            else:
                return Communicates.OSCILLOGRAM_ERROR_MESSAGE

        t_max = get_integer(t_max_text)

        if t_max is not None:
            if (0 < t_max <= 44000) and self.oscillogram_t_min < t_max:
                self.oscillogram_t_max = t_max
            else:
                return Communicates.OSCILLOGRAM_ERROR_MESSAGE

        return

    def set_spectrogram_params(self, chunks_per_ftt_text, f_min_text, f_max_text):
        chunks_per_ftt = get_integer(chunks_per_ftt_text)

        if chunks_per_ftt is not None:
            if 0 < chunks_per_ftt < 43:
                self.spectrogram_chunks_per_ftt = chunks_per_ftt
            else:
                return Communicates.SPECTROGRAM_ERROR_MESSAGE

        f_min = get_integer(f_min_text)

        if f_min is not None:
            if 0 <= f_min < 22000:
                self.spectrogram_f_min = f_min
            else:
                return Communicates.FREQUENCY_RANGE_ERROR_MESSAGE

        f_max = get_integer(f_max_text)

        if f_max is not None:
            if f_max > self.spectrogram_f_min and (0 < f_max <= 22000):
                self.spectrogram_f_max = f_max
            else:
                return Communicates.FREQUENCY_RANGE_ERROR_MESSAGE

        return
