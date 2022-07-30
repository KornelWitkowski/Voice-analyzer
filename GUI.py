from PyQt5.QtWidgets import (QMainWindow, QFrame, QPushButton, QSlider, QGridLayout,
                             QLabel, QWidget, QLineEdit, QRadioButton, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from scipy.fft import fft

from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from voice_recording_thread import voice_recording_loop
from Settings import Settings, Filters

import numpy as np
import threading


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setGeometry(300, 300, 1200, 800)
        self.setWindowTitle("Voice analyzer")

        self.frame = QFrame(self)
        self.layout = QGridLayout()
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)

        self.previewBtn = QPushButton(text='Preview')
        self.previewBtn.setFixedSize(315, 50)
        self.previewBtn.clicked.connect(self.push_preview_btn)
        self.layout.addWidget(self.previewBtn, *(0, 0, 1, 2))

        self.listenBtn = QPushButton(text='Listen')
        self.listenBtn.setFixedSize(150, 50)
        self.listenBtn.clicked.connect(self.push_listen_btn)
        self.layout.addWidget(self.listenBtn, *(0, 2, 1, 1))

        self.recordBtn = QPushButton(text='Record')
        self.recordBtn.setFixedSize(150, 50)
        self.recordBtn.clicked.connect(self.push_record_btn)
        self.layout.addWidget(self.recordBtn, *(1, 0, 1, 1))

        self.stopBtn = QPushButton(text='Stop')
        self.stopBtn.setFixedSize(150, 50)
        self.stopBtn.clicked.connect(self.push_stop_btn)
        self.layout.addWidget(self.stopBtn, *(1, 1, 1, 1))

        self.playBtn = QPushButton(text='Play')
        self.playBtn.setFixedSize(150, 50)
        self.playBtn.clicked.connect(self.push_play_btn)
        self.layout.addWidget(self.playBtn, *(1, 2, 1, 1))

        self.labelVolume = QLabel("Volume")
        self.labelVolume.setFont(QFont('Arial', 11))
        self.layout.addWidget(self.labelVolume, *(3, 0, 1, 1))
        self.sliderVolume = QSlider(Qt.Horizontal)
        self.layout.addWidget(self.sliderVolume, *(4, 0, 1, 3))
        self.sliderVolume.setMinimum(0)
        self.sliderVolume.setMaximum(100)
        self.sliderVolume.setValue(50)
        self.sliderVolume.setFocusPolicy(Qt.StrongFocus)
        self.sliderVolume.setTickPosition(QSlider.TicksBothSides)
        self.sliderVolume.setTickInterval(5)
        self.sliderVolume.setSingleStep(0)
        self.sliderVolume.valueChanged.connect(self.change_volume_slider)

        self.settings = Settings()

        self.diagrams_settings_group = DiagramsSettingsGroup(self)
        self.layout.addWidget(self.diagrams_settings_group, *(5, 0, 6, 3))

        self.filter_group = FilterGroup(self)
        self.layout.addWidget(self.filter_group, *(12, 0, 6, 3))

        self.myFig = OscillogramAndSpectrogramCanvas(self, [])
        self.layout.addWidget(self.myFig, *(0, 10, 30, 3))

        voice_recording = threading.Thread(name='voice_recording',
                                           target=voice_recording_loop,
                                           daemon=True,
                                           args=(self.add_data_callback, self))

        voice_recording.start()
        self.show()
        return

    def set_oscillogram_params(self, t_min, t_max):
        return self.settings.set_oscillogram_params(t_min, t_max)

    def set_spectrogram_params(self, chunks_per_fft, f_min, f_max):
        return self.settings.set_spectrogram_params(chunks_per_fft, f_min, f_max)

    def add_data_callback(self, value):
        self.myFig.add_data(value)
        return

    def change_volume_slider(self):
        self.settings.volume = self.sliderVolume.value()

    def push_preview_btn(self):
        self.settings.preview = not self.settings.preview
        return

    def push_listen_btn(self):

        if self.settings.listen:
            self.listenBtn.setStyleSheet("background-color: none")
        else:
            self.listenBtn.setStyleSheet("background-color: red")

        self.settings.listen = not self.settings.listen

        return

    def push_record_btn(self):
        self.settings.record = True
        self.recordBtn.setStyleSheet("background-color: red")
        return

    def push_stop_btn(self):
        self.settings.stop = True
        return

    def push_play_btn(self):
        self.settings.play = True
        self.playBtn.setStyleSheet("background-color: red")
        return


class FilterGroup(QWidget):
    def __init__(self, main_window):
        QWidget.__init__(self)
        self.main_window = main_window
        self.lay = QGridLayout(self)

        self.labelFFT = QLabel("FIR filters")
        self.labelFFT.setFont(QFont('Arial', 14))
        self.lay.addWidget(self.labelFFT, *(0, 0, 1, 3))

        self.labelBlank = QLabel("")
        self.labelBlank.setFont(QFont('Arial', 12))
        self.lay.addWidget(self.labelBlank, *(1, 0, 1, 3))

        self.radiobutton = QRadioButton("No filter")
        self.radiobutton.setChecked(True)
        self.radiobutton.filter_enum = 0
        self.radiobutton.toggled.connect(self.change_filter_radiobtn)
        self.lay.addWidget(self.radiobutton, 3, 0, 1, 1)

        self.radiobutton = QRadioButton("Low-pass")
        self.radiobutton.setChecked(False)
        self.radiobutton.filter_enum = Filters.LOW_PASS
        self.radiobutton.toggled.connect(self.change_filter_radiobtn)
        self.lay.addWidget(self.radiobutton, 3, 1, 1, 2)

        self.radiobutton = QRadioButton("Band-pass")
        self.radiobutton.setChecked(False)
        self.radiobutton.filter_enum = Filters.BAND_PASS
        self.radiobutton.toggled.connect(self.change_filter_radiobtn)
        self.lay.addWidget(self.radiobutton, 3, 3, 1, 2)

        self.radiobutton = QRadioButton("High-pass")
        self.radiobutton.setChecked(False)
        self.radiobutton.filter_enum = Filters.HIGH_PASS
        self.radiobutton.toggled.connect(self.change_filter_radiobtn)
        self.lay.addWidget(self.radiobutton, 4, 1, 1, 2)

        self.radiobutton = QRadioButton("Band-stop")
        self.radiobutton.setChecked(False)
        self.radiobutton.filter_enum = Filters.BAND_STOP
        self.radiobutton.toggled.connect(self.change_filter_radiobtn)
        self.lay.addWidget(self.radiobutton, 4, 3, 1, 2)

        self.labelBlank = QLabel("")
        self.labelBlank.setFont(QFont('Arial', 14))
        self.lay.addWidget(self.labelBlank, *(5, 0, 1, 3))

        self.labelBlank = QLabel("")
        self.labelBlank.setFont(QFont('Arial', 12))
        self.lay.addWidget(self.labelBlank, *(10, 0, 1, 3))

        self.labelFLow = QLabel("f <sub> T </sub> / f <sub> low </sub>")
        self.labelFLow.setFont(QFont('Arial', 11))
        self.labelFLow.setAlignment(Qt.AlignRight)
        self.lay.addWidget(self.labelFLow, *(11, 0, 1, 1))
        self.textboxFLow = QLineEdit(self)
        self.textboxFLow.setFixedSize(100, 25)
        self.lay.addWidget(self.textboxFLow, *(11, 1, 1, 1))

        self.labelFHigh = QLabel("f <sub> high </sub>")
        self.labelFHigh.setFont(QFont('Arial', 11))
        self.labelFHigh.setAlignment(Qt.AlignRight)
        self.lay.addWidget(self.labelFHigh, *(12, 0, 1, 1))
        self.textboxFHigh = QLineEdit(self)
        self.textboxFHigh.setFixedSize(100, 25)
        self.lay.addWidget(self.textboxFHigh, *(12, 1, 1, 1))

        self.setFilterBtn = QPushButton(text='Set')
        self.setFilterBtn.setFixedSize(100, 30)
        self.setFilterBtn.clicked.connect(self.set_filter_btn)
        self.lay.addWidget(self.setFilterBtn, *(14, 1))

    def change_filter_radiobtn(self):
        selected_button = self.sender().filter_enum
        self.main_window.settings.filters.filter_type_enum = selected_button

    def set_filter_btn(self):
        error_mess = self.main_window.settings.filters.set_filter_frequencies(self.textboxFLow.text(),
                                                                              self.textboxFHigh.text())
        if error_mess:
            QMessageBox.about(self.main_window, "Error", error_mess)

        error_mess = self.main_window.settings.filters.create_filter()

        if error_mess:
            QMessageBox.about(self.main_window, "Error", error_mess)

        return


class DiagramsSettingsGroup(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent=parent)
        self.main_window = parent
        self.layout = QGridLayout(self)

        self.labelOsc = QLabel("Oscillogram")
        self.labelOsc.setFont(QFont('Arial', 14))
        self.layout.addWidget(self.labelOsc, *(0, 0, 1, 4))

        self.labelBlank = QLabel("")
        self.labelBlank.setFont(QFont('Arial', 14))
        self.layout.addWidget(self.labelBlank, *(1, 0, 1, 4))

        self.labelTimeMin = QLabel("t <sub>min</sub>")
        self.labelTimeMin.setFont(QFont('Arial', 11))
        self.labelTimeMin.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.labelTimeMin, *(2, 0, 1, 1))
        self.textbox_t_min = QLineEdit(self)
        self.textbox_t_min.setFixedSize(100, 25)
        self.layout.addWidget(self.textbox_t_min, *(2, 1, 1, 1))

        self.labelTimeMax = QLabel("t <sub> max </sub>")
        self.labelTimeMax.setFont(QFont('Arial', 11))
        self.labelTimeMax.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.labelTimeMax, *(3, 0, 1, 1))
        self.textbox_t_max = QLineEdit(self)
        self.textbox_t_max.setFixedSize(100, 25)
        self.layout.addWidget(self.textbox_t_max, *(3, 1, 1, 1))

        self.labelFFT = QLabel("FFT")
        self.labelFFT.setFont(QFont('Arial', 14))
        self.layout.addWidget(self.labelFFT, *(0, 5, 1, 4))

        self.labelAveraging = QLabel("Chunks per FFT")
        self.labelAveraging.setFont(QFont('Arial', 11))
        self.labelAveraging.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.labelAveraging, *(2, 5, 1, 1))
        self.textboxAveragesFFT = QLineEdit(self)
        self.textboxAveragesFFT.setFixedSize(100, 25)
        self.layout.addWidget(self.textboxAveragesFFT, *(2, 6, 1, 1))

        self.labelFmin = QLabel("f <sub>min</sub>")
        self.labelFmin.setFont(QFont('Arial', 11))
        self.labelFmin.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.labelFmin, *(3, 5, 1, 1))
        self.textboxFmin = QLineEdit(self)
        self.textboxFmin.setFixedSize(100, 25)
        self.layout.addWidget(self.textboxFmin, *(3, 6, 1, 1))

        self.labelFmax = QLabel("f <sub>max</sub>")
        self.labelFmax.setFont(QFont('Arial', 11))
        self.labelFmax.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.labelFmax, *(4, 5, 1, 1))
        self.textboxFmax = QLineEdit(self)
        self.textboxFmax.setFixedSize(100, 25)
        self.layout.addWidget(self.textboxFmax, *(4, 6, 1, 1))

        self.setBtn = QPushButton(text='Set')
        self.setBtn.setFixedSize(100, 30)
        self.setBtn.clicked.connect(self.set_oscillogram_and_spectogram)
        self.layout.addWidget(self.setBtn, *(5, 1))

    def set_oscillogram_and_spectogram(self):
        error_mess = self.main_window.set_oscillogram_params(self.textbox_t_min.text(),
                                                             self.textbox_t_max.text())

        if error_mess:
            QMessageBox.about(self.main_window, "Error", error_mess)

        error_mess = self.main_window.set_spectrogram_params(self.textboxAveragesFFT.text(),
                                                             self.textboxFmin.text(),
                                                             self.textboxFmax.text())

        if error_mess:
            QMessageBox.about(self.main_window, "Error", error_mess)

        data = self.main_window.myFig.y
        self.main_window.myFig.__del__()
        self.main_window.myFig = OscillogramAndSpectrogramCanvas(self.main_window, data)
        self.main_window.layout.addWidget(self.main_window.myFig, *(0, 10, 30, 3))

        return


class OscillogramAndSpectrogramCanvas(FigureCanvas, TimedAnimation):
    def __init__(self, main_window, data):
        self.addedData = []
        self.main_window = main_window
        self.chunk = self.main_window.settings.CHUNK
        self.rate = self.main_window.settings.RATE

        self.xlim = 44000
        self.n = np.linspace(0, self.xlim - 1, self.xlim)

        self.y = self.n * 0.0 if data == [] else data

        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax1 = self.fig.add_subplot(211)

        self.ax1.set_xlabel('Time', fontsize=14)
        self.ax1.set_ylabel('Amplitude', fontsize=14)
        self.ax1.set_title('Oscillogram', fontsize=20)
        self.line1 = Line2D([], [], color='blue', linewidth=1)
        self.ax1.add_line(self.line1)
        self.ax1.set_xlim(main_window.settings.oscillogram_t_min, main_window.settings.oscillogram_t_max - 1)
        self.ax1.set_ylim(-10000, 10000)

        self.xlimFFT = 6000

        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_xlabel('Frequency [Hz]', fontsize=14)
        self.ax2.set_ylabel('Amplitude', fontsize=14)
        self.ax2.set_title('FFT', fontsize=20)
        self.line2 = Line2D([], [], color='red', linewidth=1)
        self.ax2.add_line(self.line2)
        self.ax2.set_xlim(main_window.settings.spectrogram_f_min, main_window.settings.spectrogram_f_max - 1)
        self.ax2.set_ylim(-20, 600)

        self.fig.tight_layout()

        FigureCanvas.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval = 10, blit = True)
        return

    def new_frame_seq(self):
        return iter(range(self.n.size))

    def _init_draw(self):
        lines = [self.line1]
        for line in lines:
            line.set_data([], [])
        lines = [self.line2]
        for line in lines:
            line.set_data([], [])
        return

    def add_data(self, value):
        self.addedData = list(value)
        return

    def _step(self, *args):

        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            self.abc += 1
            TimedAnimation._stop(self)
            pass
        return

    def _draw_frame(self, framedata):
        margin = 2

        avg = self.main_window.settings.spectrogram_chunks_per_ftt

        while len(self.addedData) > 0:
            self.y = np.roll(self.y, 1)
            self.y[1] = self.addedData[0]
            del(self.addedData[0])

        self.z = abs(fft(self.y[0:(avg*self.chunk)]))/len(self.y[0:(avg*self.chunk)])

        self.line1.set_data(self.n[0: self.n.size - margin], self.y[0: self.n.size - margin])
        self.line2.set_data(self.rate/(avg*self.chunk)*self.n[0: (avg*self.chunk) - margin],
                            self.z[0: (avg*self.chunk) - margin])

        self._drawn_artists = [self.line1, self.line2]

        return


