from enum import Enum


class Communicates:
    OSCILLOGRAM_ERROR_MESSAGE = "Invalid minimal time. Should be an integer from 0 to 44000."
    SPECTROGRAM_ERROR_MESSAGE = "Invalid 'chunk per FFT' value. Should be an integer from 1 to 42."
    FREQUENCY_RANGE_ERROR_MESSAGE = "Invalid value f min. Set an integer from 0 to 22000."
    CUTOFF_FREQUENCY_ERROR_MESSAGE = "Incorrect filter cutoff frequency."
    FILTER_FREQUENCY_ERROR_MESSAGE = "Invalid filter frequency value."


class Filters(Enum):
    NO_FILTER = 0
    LOW_PASS = 1
    HIGH_PASS = 2
    BAND_PASS = 3
    BAND_STOP = 4
