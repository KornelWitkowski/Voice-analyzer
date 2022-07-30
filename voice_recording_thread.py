import pyaudio
import wave
import numpy as np

from PyQt5.QtCore import pyqtSignal, QObject


class Communicate(QObject):
    data_signal = pyqtSignal(np.ndarray)


def get_recording(stream, chunk, volume, fir_filter):
    recording = np.int16(volume/50 * np.fromstring(stream.read(chunk), dtype=np.int16))

    if fir_filter is not None:
        recording = np.convolve(recording, fir_filter, mode='same')
    return recording


def emit_recording(stream, player, source, settings, emit_to_preview=True, emit_to_player=False):
    recording = get_recording(stream,
                              settings.CHUNK,
                              settings.volume,
                              settings.filters.fir_filter)

    if emit_to_preview and settings.preview:
        source.data_signal.emit(recording)
    if emit_to_player and settings.listen:
        player.write(np.int16(recording), settings.CHUNK)

    return recording

def save_recording_to_file(p, frames, rate):
    file = wave.open("recording.wav", 'wb')
    file.setnchannels(1)
    file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    file.setframerate(rate)
    file.writeframes(b''.join(frames))
    file.close()
    return


def record_voice(p, stream, player, source, main_window):
    frames = []
    seconds = 0
    seconds_per_frame = main_window.settings.CHUNK / main_window.settings.RATE

    while not main_window.settings.stop and seconds < 30:
        recorded_voice = emit_recording(stream, player, source, main_window.settings)
        frames.append(np.int16(recorded_voice))
        seconds = seconds + seconds_per_frame

    save_recording_to_file(p, frames, main_window.settings.RATE)

    main_window.settings.record = False
    main_window.recordBtn.setStyleSheet("background-color: none")

    return


def play_recording_from_file(main_window, source):
    filename = 'recording.wav'
    wf = wave.open(filename, 'rb')
    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=1,
                    rate=wf.getframerate(),
                    output=True)

    frame = wf.readframes(main_window.settings.CHUNK)

    while len(frame) > 0 and not main_window.settings.stop:
        if main_window.settings.preview:
            source.data_signal.emit(np.frombuffer(frame, dtype=np.int16))
        stream.write(frame)
        frame = wf.readframes(main_window.settings.CHUNK)

    stream.close()
    p.terminate()

    main_window.settings.stop = False
    main_window.settings.play = False
    main_window.playBtn.setStyleSheet("background-color: none")

    return


def start_new_stream(rate, chunk):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)
    player = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    output=True,
                    frames_per_buffer=chunk)

    return p, stream, player


def voice_recording_loop(add_data_callback, main_window):

    source = Communicate()
    source.data_signal.connect(add_data_callback)

    (p, stream, player) = start_new_stream(main_window.settings.RATE, main_window.settings.CHUNK)
    new_stream = False

    while True:

        if new_stream:
            (p, stream, player) = start_new_stream(main_window.settings.RATE, main_window.settings.CHUNK)
            new_stream = False

        if main_window.settings.preview or main_window.settings.listen:
            emit_recording(stream, player, source, main_window.settings, emit_to_preview=True, emit_to_player=True)

        if main_window.settings.record:
            record_voice(p, stream, player, source, main_window)

        if main_window.settings.play:
            stream.close()
            p.terminate()
            play_recording_from_file(main_window, source)
            new_stream = True
