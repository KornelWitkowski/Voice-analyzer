import sys

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import wave
import numpy as np
import pyaudio
from scipy.fft import fft, ifft
from scipy import signal

from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time
import threading


volume = 50
avg = 5
fmin = 0
fmax = 22000
rate = 44000
chunk = 1023
tmin = 0
tmax = 44000

Order = 2
Ripple = 20

filterKind = "FIR"
filterType = "butter"

filterRadio = "None"

record = False
preview = False
stop = False
play = False
listen = False

chosenFilter = "None"


class CustomMainWindow(QMainWindow):
    def __init__(self):
        super(CustomMainWindow, self).__init__()


        self.setGeometry(300, 300, 1200, 800)
        self.setWindowTitle("Voice analyzer")

        self.frame = QFrame(self)
        self.layout = QGridLayout()
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)

        self.previewBtn = QPushButton(text='Preview')
        self.previewBtn.setFixedSize(315, 50)
        self.previewBtn.clicked.connect(self.previewBtnAction)
        self.layout.addWidget(self.previewBtn, *(0, 0, 1, 2))

        self.listenBtn = QPushButton(text='Listen')
        self.listenBtn.setFixedSize(150, 50)
        self.listenBtn.clicked.connect(self.listenBtnAction)
        self.layout.addWidget(self.listenBtn, *(0, 2, 1, 1))

        self.recordBtn = QPushButton(text='Record')
        self.recordBtn.setFixedSize(150, 50)
        self.recordBtn.clicked.connect(self.recordBtnAction)
        self.layout.addWidget(self.recordBtn, *(1, 0, 1, 1))

        self.stopBtn = QPushButton(text = 'Stop')
        self.stopBtn.setFixedSize(150, 50)
        self.stopBtn.clicked.connect(self.stopBtnAction)
        self.layout.addWidget(self.stopBtn, *(1, 1, 1, 1))

        self.playBtn = QPushButton(text = 'Play')
        self.playBtn.setFixedSize(150, 50)
        self.playBtn.clicked.connect(self.playBtnAction)
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
        self.sliderVolume.valueChanged.connect(self.volumeSliderChange)

        self.panelDiagram = diagramPanelGroup(self)
        self.layout.addWidget(self.panelDiagram, *(5, 0, 6, 3))

        self.panelFilter = filterGroup(self)
        self.layout.addWidget(self.panelFilter, *(12, 0, 6, 3))

        self.myFig = CustomFigCanvas([])
        self.layout.addWidget(self.myFig, *(0, 10, 30, 3))

        myDataLoop = threading.Thread(name='myDataLoop', target=dataSendLoop,
                                      daemon=True, args=(self.addData_callbackFunc, self))

        myDataLoop.start()
        self.show()
        return

    def volumeSliderChange(self):
        global volume
        volume = self.sliderVolume.value()

    def previewBtnAction(self):
        global preview
        preview = not preview
        return

    def listenBtnAction(self):
        global listen

        if listen:
            self.listenBtn.setStyleSheet("background-color: none")
        else:
            self.listenBtn.setStyleSheet("background-color: red")

        listen = not listen

        return

    def recordBtnAction(self):
        global record
        record = True
        self.recordBtn.setStyleSheet("background-color: red")
        return

    def stopBtnAction(self):
        global stop
        stop = True
        return

    def playBtnAction(self):
        global play
        play = True
        self.playBtn.setStyleSheet("background-color: red")
        return

    def addData_callbackFunc(self, value):
        self.myFig.addData(value)
        return


class filterGroup(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent=parent)
        self.lay = QGridLayout(self)

        self.labelFFT = QLabel("Filters")
        self.labelFFT.setFont(QFont('Arial', 14))
        self.lay.addWidget(self.labelFFT, *(0, 0, 1, 3))

        self.labelBlank = QLabel("")
        self.labelBlank.setFont(QFont('Arial', 12))
        self.lay.addWidget(self.labelBlank, *(1, 0, 1, 3))

        # self.labelFIR = QLabel("FIR")
        # self.labelFIR.setFont(QFont('Arial', 12))
        # self.lay.addWidget(self.labelFIR, *(2, 0, 1, 3))

        self.radiobutton = QRadioButton("No filter")
        self.radiobutton.setChecked(True)
        self.radiobutton.filter = "None"
        self.radiobutton.toggled.connect(self.filterRadioBtn)
        self.lay.addWidget(self.radiobutton, 3, 0, 1, 1)

        self.radiobutton = QRadioButton("Low-pass")
        self.radiobutton.setChecked(False)
        self.radiobutton.filter = "Low-pass"
        self.radiobutton.toggled.connect(self.filterRadioBtn)
        self.lay.addWidget(self.radiobutton, 3, 1, 1, 2)

        self.radiobutton = QRadioButton("Band-pass")
        self.radiobutton.setChecked(False)
        self.radiobutton.filter = "Band-pass"
        self.radiobutton.toggled.connect(self.filterRadioBtn)
        self.lay.addWidget(self.radiobutton, 3, 3, 1, 2)

        self.radiobutton = QRadioButton("High-pass")
        self.radiobutton.setChecked(False)
        self.radiobutton.filter = "High-pass"
        self.radiobutton.toggled.connect(self.filterRadioBtn)
        self.lay.addWidget(self.radiobutton, 4, 1, 1, 2)

        self.radiobutton = QRadioButton("Band-stop")
        self.radiobutton.setChecked(False)
        self.radiobutton.filter = "Band-stop"
        self.radiobutton.toggled.connect(self.filterRadioBtn)
        self.lay.addWidget(self.radiobutton, 4, 3, 1, 2)

        self.labelBlank = QLabel("")
        self.labelBlank.setFont(QFont('Arial', 14))
        self.lay.addWidget(self.labelBlank, *(5, 0, 1, 3))

        self.FilterKindGroup = QButtonGroup()

        self.radiobutton = QRadioButton("IIR")
        self.radiobutton.setChecked(False)
        self.radiobutton.label = "IIR"
        self.radiobutton.toggled.connect(self.filterKindRadioBtn)
        self.FilterKindGroup.addButton(self.radiobutton)
        self.lay.addWidget(self.radiobutton, 8, 0, 1, 2)

        self.radiobutton = QRadioButton("FIR")
        self.radiobutton.setChecked(True)
        self.radiobutton.label = "FIR"
        self.radiobutton.toggled.connect(self.filterKindRadioBtn)
        self.FilterKindGroup.addButton(self.radiobutton)
        self.lay.addWidget(self.radiobutton, 9, 0, 1, 2)

        self.IIRgroup = QButtonGroup()

        self.radiobutton = QRadioButton("Butterworth")
        self.radiobutton.setChecked(True)
        self.radiobutton.label = "butter"
        self.radiobutton.toggled.connect(self.IIRfilterRadioBtn)
        self.IIRgroup.addButton(self.radiobutton)
        self.lay.addWidget(self.radiobutton, 8, 1, 1, 2)

        self.radiobutton = QRadioButton("Bessel")
        self.radiobutton.setChecked(False)
        self.radiobutton.label = "bessel"
        self.radiobutton.toggled.connect(self.IIRfilterRadioBtn)
        self.IIRgroup.addButton(self.radiobutton)
        self.lay.addWidget(self.radiobutton, 8, 3, 1, 2)

        self.radiobutton = QRadioButton("Chebyshev I")
        self.radiobutton.setChecked(False)
        self.radiobutton.label = "cheby1"
        self.radiobutton.toggled.connect(self.IIRfilterRadioBtn)
        self.IIRgroup.addButton(self.radiobutton)
        self.lay.addWidget(self.radiobutton, 9, 1, 1, 2)

        self.radiobutton = QRadioButton("Chebyshev II")
        self.radiobutton.setChecked(False)
        self.radiobutton.label = "cheby2"
        self.radiobutton.toggled.connect(self.IIRfilterRadioBtn)
        self.IIRgroup.addButton(self.radiobutton)
        self.lay.addWidget(self.radiobutton, 9, 3, 1, 2)

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

        self.labelFilterOrder = QLabel("Order")
        self.labelFilterOrder.setFont(QFont('Arial', 9))
        self.labelFilterOrder.setAlignment(Qt.AlignRight)
        self.lay.addWidget(self.labelFilterOrder, *(11, 3, 1, 1))
        self.textboxFilterOrder = QLineEdit(self)
        self.textboxFilterOrder.setFixedSize(100, 25)
        self.lay.addWidget(self.textboxFilterOrder, *(11, 4, 1, 1))

        self.labelFilterRipple = QLabel("Ripple")
        self.labelFilterRipple.setFont(QFont('Arial', 9))
        self.labelFilterRipple.setAlignment(Qt.AlignRight)
        self.lay.addWidget(self.labelFilterRipple, *(12, 3, 1, 1))
        self.textboxFilterRipple = QLineEdit(self)
        self.textboxFilterRipple.setFixedSize(100, 25)
        self.lay.addWidget(self.textboxFilterRipple, *(12, 4, 1, 1))

        self.setFilterBtn = QPushButton(text='Set')
        self.setFilterBtn.setFixedSize(100, 30)
        self.setFilterBtn.clicked.connect(self.setFilterAction)
        self.lay.addWidget(self.setFilterBtn, *(14, 1))

        self.setFilterBtn = QPushButton(text='Filter preview')
        self.setFilterBtn.setFixedSize(100, 30)
        self.setFilterBtn.clicked.connect(self.setFilterPreview)
        self.lay.addWidget(self.setFilterBtn, *(14, 4))


    def setFilterPreview(self):
        self.w = FilterFrequencyResponsePreview()
        self.w.show()

    def filterKindRadioBtn(self):
        global filterKind
        radioButton = self.sender()

        if radioButton.isChecked():
            if radioButton.label == "IIR":
                filterKind = "IIR"
            else:
                filterKind = "FIR"

    def IIRfilterRadioBtn(self):
        global filterType
        radioButton = self.sender()

        if radioButton.isChecked():
            filterType = radioButton.label
        else:
            filterType = "butter"

    def filterRadioBtn(self):
        global filterRadio
        radioButton = self.sender()

        if radioButton.isChecked():
            filterRadio = radioButton.filter


    def setFilterAction(self):

        global filterRadio, chosenFilter, FIRfilter, IIRfilterA, IIRfilterB, chunk, rate, Order, Ripple, filterType, IIRfilterA_analog, IIRfilterB_analog

        FilterRipple = self.textboxFilterRipple.text()
        FilterOrder = self.textboxFilterOrder.text()

        FLowValue = self.textboxFLow.text()
        FHighValue = self.textboxFHigh.text()

        if filterRadio == "None":
            chosenFilter = "None"
            return


        if bool(FilterRipple):
            try:
                FilterRipple = int(FilterRipple)
                if 1 < FilterRipple < 100:
                    Ripple = FilterRipple
                else:
                    Ripple = False
            except:
                Ripple = False
        else:
            Ripple = False

        if bool(FilterOrder):
            try:
                FilterOrder = int(FilterOrder)
                if 0 < FilterOrder < 40:
                    Order = FilterOrder
                else:
                    Order = False
            except:
                Order = False
        else:
            Order = False


        if bool(FLowValue):
            try:
                FLowValue = int(FLowValue)
                if 0 < FLowValue < 22000:
                    FLow = FLowValue
                else:
                    FLow = False
            except:
                FLow = False
        else:
            FLow = False

        if bool(FHighValue):
            try:
                FHighValue = int(FHighValue)
                if 0 < FHighValue < 22000 and FHighValue > FLow:
                    FHigh = FHighValue
                else:
                    FHigh = False
            except:
                FHigh = False
        else:
            FHigh = False

        if filterRadio == "Low-pass" or filterRadio == "High-pass":
            if FLow == False:
                QMessageBox.about(self, "Error", "Incorrect cutoff frequency")
            elif Order == False and filterKind == "IIR":
                QMessageBox.about(self, "Error", "Incorrect filter order")
            elif Ripple == False and filterKind == "IIR":
                QMessageBox.about(self, "Error", "Invalid ripple value.")
            else:
                if filterRadio == "Low-pass":

                    chosenFilter = "Low-pass"
                    FIRfilter = signal.firwin(chunk, cutoff=FLow, fs=rate)

                    if filterType == "cheby1" or filterType == "cheby2":
                        IIRfilterB, IIRfilterA = signal.iirfilter(Order, Wn=FLow, fs=44000, btype='lowpass',
                                                                  ftype=filterType, rp=Ripple, rs=Ripple)
                        IIRfilterB_analog, IIRfilterA_analog = signal.iirfilter(Order, Wn=FLow, analog=True, btype='lowpass',
                                                                  ftype=filterType, rp=Ripple, rs=Ripple)
                    else:
                        IIRfilterB, IIRfilterA = signal.iirfilter(Order, Wn=FLow, fs=44000, btype='lowpass', ftype=filterType)
                        IIRfilterB_analog, IIRfilterA_analog = signal.iirfilter(Order, Wn=FLow, btype='lowpass', analog=True, ftype=filterType)
                else:
                    chosenFilter = "High-pass"
                    FIRfilter = signal.firwin(numtaps=chunk, cutoff=FLow, fs=rate, pass_zero=False)
                    if filterType == "cheby1" or filterType == "cheby2":
                        IIRfilterB, IIRfilterA = signal.iirfilter(Order, Wn=FLow, fs=44000, btype='highpass',
                                                                  ftype=filterType, rp=Ripple, rs=Ripple)
                        IIRfilterB_analog, IIRfilterA_analog = signal.iirfilter(Order, Wn=FLow, btype='highpass',
                                                                                analog=True, ftype=filterType, rp=Ripple, rs=Ripple)
                    else:
                        IIRfilterB, IIRfilterA = signal.iirfilter(Order, Wn=FLow, fs=44000, btype='highpass', ftype=filterType)
                        IIRfilterB_analog, IIRfilterA_analog = signal.iirfilter(Order, Wn=FLow, analog=True, btype='highpass', ftype=filterType)

        else:
            if not FLow or not FHigh:
                QMessageBox.about(self,"Error", "Invalid frequency value.")
            else:
                if filterRadio == "Band-pass":
                    chosenFilter = "Band-pass"
                    FIRfilter = signal.firwin(chunk, [FLow, FHigh], fs=rate, pass_zero=False)

                    if filterType == "cheby1" or filterType == "cheby2":
                        IIRfilterB, IIRfilterA = signal.iirfilter(Order, Wn=[FLow, FHigh], fs=44000, btype='bandpass',
                                                                  ftype=filterType, rp=Ripple, rs=Ripple)
                        IIRfilterB_analog, IIRfilterA_analog = signal.iirfilter(Order, Wn=[FLow, FHigh], analog=True, btype='bandpass',
                                                                  ftype=filterType, rp=Ripple, rs=Ripple)
                    else:
                        IIRfilterB, IIRfilterA = signal.iirfilter(Order, Wn=[FLow, FHigh], fs=44000, btype='bandpass', ftype=filterType)
                        IIRfilterB_analog, IIRfilterA_analog = signal.iirfilter(Order, Wn=[FLow, FHigh], analog=True, btype='bandpass', ftype=filterType)
                else:
                    chosenFilter = "Band-stop"
                    FIRfilter = signal.firwin(chunk, [FLow, FHigh], fs=rate)

                    if filterType == "cheby1" or filterType == "cheby2":
                        IIRfilterB, IIRfilterA = signal.iirfilter(Order, Wn=[FLow, FHigh], fs=44000, btype='bandstop',
                                                                  ftype=filterType, rp=Ripple, rs=Ripple)
                        IIRfilterB_analog, IIRfilterA_analog = signal.iirfilter(Order, Wn=[FLow, FHigh], analog=True, btype='bandstop',
                                                                  ftype=filterType, rp=Ripple, rs=Ripple)
                    else:
                        IIRfilterB, IIRfilterA = signal.iirfilter(Order, Wn=[FLow, FHigh], fs=44000, btype='bandstop',
                                                                  ftype=filterType)
                        IIRfilterB_analog, IIRfilterA_analog = signal.iirfilter(Order, Wn=[FLow, FHigh], analog=True, btype='bandstop',
                                                                  ftype=filterType)
        return


class diagramPanelGroup(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent=parent)
        lay = QGridLayout(self)

        self.labelOsc = QLabel("Oscillogram")
        self.labelOsc.setFont(QFont('Arial', 14))
        lay.addWidget(self.labelOsc, *(0, 0, 1, 4))

        self.labelBlank = QLabel("")
        self.labelBlank.setFont(QFont('Arial', 14))
        lay.addWidget(self.labelBlank, *(1, 0, 1, 4))

        self.labelTimeMin = QLabel("t <sub>min</sub>")
        self.labelTimeMin.setFont(QFont('Arial', 11))
        self.labelTimeMin.setAlignment(Qt.AlignRight)
        lay.addWidget(self.labelTimeMin, *(2, 0, 1, 1))
        self.textboxTimeMin=QLineEdit(self)
        self.textboxTimeMin.setFixedSize(100, 25)
        lay.addWidget(self.textboxTimeMin, *(2, 1, 1, 1))

        self.labelTimeMax = QLabel("t <sub> max </sub>")
        self.labelTimeMax.setFont(QFont('Arial', 11))
        self.labelTimeMax.setAlignment(Qt.AlignRight)
        lay.addWidget(self.labelTimeMax, *(3, 0, 1, 1))
        self.textboxTimeMax=QLineEdit(self)
        self.textboxTimeMax.setFixedSize(100, 25)
        lay.addWidget(self.textboxTimeMax, *(3, 1, 1, 1))

        self.labelFFT = QLabel("FFT")
        self.labelFFT.setFont(QFont('Arial', 14))
        lay.addWidget(self.labelFFT, *(0, 5, 1, 4))

        self.labelAveraging = QLabel("n")
        self.labelAveraging.setFont(QFont('Arial', 11))
        self.labelAveraging.setAlignment(Qt.AlignRight)
        lay.addWidget(self.labelAveraging, *(2, 5, 1, 1))
        self.textboxAveragesFFT=QLineEdit(self)
        self.textboxAveragesFFT.setFixedSize(100, 25)
        lay.addWidget(self.textboxAveragesFFT, *(2, 6, 1, 1))

        self.labelFmin = QLabel("f <sub>min</sub>")
        self.labelFmin.setFont(QFont('Arial', 11))
        self.labelFmin.setAlignment(Qt.AlignRight)
        lay.addWidget(self.labelFmin, *(3, 5, 1, 1))
        self.textboxFmin=QLineEdit(self)
        self.textboxFmin.setFixedSize(100, 25)
        lay.addWidget(self.textboxFmin, *(3, 6, 1, 1))

        self.labelFmax = QLabel("f <sub>max</sub>")
        self.labelFmax.setFont(QFont('Arial', 11))
        self.labelFmax.setAlignment(Qt.AlignRight)
        lay.addWidget(self.labelFmax, *(4, 5, 1, 1))
        self.textboxFmax=QLineEdit(self)
        self.textboxFmax.setFixedSize(100, 25)
        lay.addWidget(self.textboxFmax, *(4, 6, 1, 1))

        self.setBtn = QPushButton(text = 'Set')
        self.setBtn.setFixedSize(100, 30)
        self.setBtn.clicked.connect(self.setBtnAction)
        lay.addWidget(self.setBtn, *(5, 1))


    def setBtnAction(self):

        global avg, fmax, fmin, tmin, tmax

        Parent = self.parent().parent()

        tminValue = self.textboxTimeMin.text()

        if bool(tminValue):
            try:
                tminValue = int(tminValue)
                if 0 <= tminValue <= 44000:
                    tmin = tminValue
                else:
                    QMessageBox.about(Parent,"Error", "Invalid minimal time. Set an integer from 0 to 44000")
            except:
                QMessageBox.about(Parent, "Error", "Invalid minimal time. Set an integer from 0 to 44000")

        tmaxValue = self.textboxTimeMax.text()

        if bool(tmaxValue):
            try:
                tmaxValue = int(tmaxValue)
                if 100 < tmaxValue <= 44000:
                    tmax = tmaxValue
                else:
                    QMessageBox.about(Parent, "Error", "Invalid maximal time. Set an integer from 100 to 44000.")
            except:
                QMessageBox.about(Parent, "Error", "Invalid maximal time. Set an integer from 100 to 44000.")

        averagingValue = self.textboxAveragesFFT.text()

        if bool(averagingValue):
            try:
                averagingValue = int(averagingValue)
                if 0 < averagingValue < 43:
                    avg = averagingValue
                else:
                    QMessageBox.about(Parent, "Error", "Invalid n value. Set an integer from 1 to 42.")
            except:
                QMessageBox.about(Parent, "Error", "Invalid n value. Set an integer from 1 to 42.")

        fminValue = self.textboxFmin.text()

        if bool(fminValue):
            try:
                fminValue = int(fminValue)

                if 0 <= fminValue < 22000:
                    fmin = fminValue
                else:
                    QMessageBox.about(Parent, "Error", "Invalid value f min. Set an integer from 0 to 22000.")
            except:
                QMessageBox.about(Parent, "Error", "Invalid value f min. Set an integer from 0 to 22000.")

        fmaxValue = self.textboxFmax.text()

        if bool(fmaxValue):
            try:
                fmaxValue = int(fmaxValue)

                if 0 < fmaxValue <= 22000 and fmaxValue > fmin:
                    fmax = fmaxValue
                else:
                    QMessageBox.about(Parent, "Error", "Invalid f max value. Set integer from 1 to 22000.")
            except:
                QMessageBox.about(Parent, "Error", "Invalid f max value. Set integer from 1 to 22000.")

        data = Parent.myFig.y
        Parent.myFig.__del__()
        Parent.myFig = CustomFigCanvas(data)
        Parent.layout.addWidget(Parent.myFig, *(0, 10, 30, 3))

        return


class FilterFrequencyResponsePreview(QWidget):

    def __init__(self):
        super().__init__()

        global IIRfilterB, IIRfilterA

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        w, h = signal.freqs(IIRfilterB_analog, IIRfilterA_analog, 10000)

        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_xlabel('Frequency [Hz]', fontsize=14)
        self.ax1.set_ylabel('Amplitude [dB]', fontsize=14)
        self.ax1.set_title('Filter frequency response', fontsize=20)

        self.ax1.margins(0, 0.1)
        self.ax1.grid(which='both', axis='both')
        self.ax1.plot(w,  20 * np.log10(abs(h)))
        self.ax1.set_xlim(1, 22000)
        self.ax1.set_ylim(-200, 10)

        self.obrazek = FigureCanvas(self.fig)
        self.layout.addWidget(self.obrazek)


class CustomFigCanvas(FigureCanvas, TimedAnimation):
    def __init__(self, data):
        self.addedData = []

        # The data
        self.xlim = 44000
        self.n = np.linspace(0, self.xlim - 1, self.xlim)

        if data == []:
            self.y = (self.n * 0.0)
        else:
            self.y = data

        self.fig = Figure(figsize=(5,5), dpi=100)
        self.ax1 = self.fig.add_subplot(211)

        self.ax1.set_xlabel('Time', fontsize=14)
        self.ax1.set_ylabel('Amplitude', fontsize=14)
        self.ax1.set_title('Oscillogram', fontsize=20)
        self.line1 = Line2D([], [], color='blue', linewidth=1)
        self.ax1.add_line(self.line1)
        self.ax1.set_xlim(tmin, tmax - 1)
        self.ax1.set_ylim(-10000, 10000)

        self.xlimFFT = 6000

        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_xlabel('Frequency [Hz]', fontsize=14)
        self.ax2.set_ylabel('Amplitude', fontsize=14)
        self.ax2.set_title('FFT', fontsize=20)
        self.line2 = Line2D([], [], color='red', linewidth=1)
        self.ax2.add_line(self.line2)
        self.ax2.set_xlim(fmin, fmax - 1)
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

    def addData(self, value):
        self.addedData = list(value)
        return

    def _step(self, *args):

        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            self.abc += 1
            print(str(self.abc))
            TimedAnimation._stop(self)
            pass
        return

    def _draw_frame(self, framedata):
        margin = 2

        global avg

        while len(self.addedData) > 0:
            self.y = np.roll(self.y, 1)
            self.y[1] = self.addedData[0]
            del(self.addedData[0])

        self.z = abs(fft(self.y[0:(avg*chunk)]))/len(self.y[0:(avg*chunk)])

        self.line1.set_data(self.n[0: self.n.size - margin], self.y[0: self.n.size - margin])
        self.line2.set_data(44000/(avg*chunk)*self.n[0: (avg*chunk) - margin], self.z[0: (avg*chunk) - margin])

        self._drawn_artists = [self.line1, self.line2]

        return


class Communicate(QObject):
    data_signal = pyqtSignal(np.ndarray)


def dataSendLoop(addData_callbackFunc, MainWindow):

    mySrc = Communicate()
    mySrc.data_signal.connect(addData_callbackFunc)

    global record, preview, play, stop, rate, chunk, volume, filterKind, IIRfilterB, IIRfilterA
    new_stream = True

    while True:



        if new_stream:

            p=pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk)
            new_stream = False
            player = p.open(format=pyaudio.paInt16, channels=1, rate=rate, output=True,
                            frames_per_buffer=chunk)

        if preview or listen:

            recording = np.fromstring(stream.read(chunk), dtype=np.int16)
            recording = np.int16(volume/50*recording)

            if chosenFilter != "None":



                if filterKind == "FIR":
                    recording = np.convolve(recording, FIRfilter, mode='same')
                else:

                        zi = signal.lfilter_zi(IIRfilterB, IIRfilterA)
                        z, _ = signal.lfilter(IIRfilterB, IIRfilterA, recording, zi=zi * recording[0])
                        z2, _ = signal.lfilter(IIRfilterB, IIRfilterA, z, zi=zi * z[0])
                        recording = signal.filtfilt(IIRfilterB, IIRfilterA, recording)

            if preview:
                mySrc.data_signal.emit(recording)
            if listen:
                player.write(np.int16(recording), chunk)

        else:
            time.sleep(0.1)

        if record:
            time.sleep(0.2)
            frames = []
            seconds = 0

            while not stop and seconds < 30:

                data = np.fromstring(stream.read(chunk),dtype= np.int16)
                data = np.int16(volume/50*data)

                if chosenFilter != "None":
                    if filterKind == "FIR":
                        recording = np.convolve(recording, FIRfilter, mode='same')
                    else:

                        zi = signal.lfilter_zi(IIRfilterB, IIRfilterA)
                        z, _ = signal.lfilter(IIRfilterB, IIRfilterA, recording, zi=zi * recording[0])
                        z2, _ = signal.lfilter(IIRfilterB, IIRfilterA, z, zi=zi * z[0])
                        recording = signal.filtfilt(IIRfilterB, IIRfilterA, recording)
                if preview:
                    mySrc.data_signal.emit(data)
                frames.append(np.int16(data))
                seconds = seconds + chunk/rate

            wf = wave.open("output.wav", 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
            wf.close()

            record = False
            MainWindow.recordBtn.setStyleSheet("background-color: none")

        if play:

            time.sleep(0.2)
            stream.close()
            p.terminate()
            filename = 'output.wav'
            wf = wave.open(filename, 'rb')

            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=1,
                            rate=wf.getframerate(),
                            output=True)

            data = wf.readframes(chunk)

            while len(data)>0 and not stop:
                if preview:
                    mySrc.data_signal.emit(np.frombuffer(data, dtype=np.int16))
                stream.write(data)
                data = wf.readframes(chunk)

            stream.close()
            p.terminate()

            stop = False
            play = False
            new_stream = True
            MainWindow.playBtn.setStyleSheet("background-color: none")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    QApplication.setStyle(QStyleFactory.create('Plastique'))
    myGUI = CustomMainWindow()
    sys.exit(app.exec_())
