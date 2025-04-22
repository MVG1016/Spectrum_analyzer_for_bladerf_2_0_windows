from bladerf import _bladerf
import numpy as np
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
import sys


class SpectrumAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- Конфигурация SDR ---
        self.sdr = _bladerf.BladeRF()
        self.rx = self.sdr.Channel(_bladerf.CHANNEL_RX(0))

        # --- Параметры ---
        self.START_FREQ = 100e6
        self.END_FREQ = 5700e6
        self.SPAN = 20e6
        self.STEP = 10e6
        self.SAMPLE_RATE = 40e6
        self.NUM_SAMPLES = 8192
        self.GAIN = 18
        self.WINDOW_TYPE = 'hann'

        # --- Калибровочная таблица ---
        self.calibration_table = {
            100e6:  {"freq_offset": 300e6, "power_offset": 0},
            500e6:  {"freq_offset": 300e6, "power_offset": 0},
            1000e6: {"freq_offset": 300e6, "power_offset": 0},
            2000e6: {"freq_offset": 300e6, "power_offset": 0},
            2400e6: {"freq_offset": 300e6, "power_offset": 0},
            3000e6: {"freq_offset": 300e6, "power_offset": 0},
            4000e6: {"freq_offset": 300e6, "power_offset": 0},
            5000e6: {"freq_offset": 300e6, "power_offset": 0},
            5800e6: {"freq_offset": 300e6, "power_offset": 0},
        }

        # --- Инициализация SDR ---
        self.init_sdr()

        # --- Оконная функция ---
        self.window = self.get_window(self.WINDOW_TYPE)
        self.window_correction = np.mean(self.window ** 2)

        # --- GUI ---
        self.init_ui()

        # --- Таймер ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_spectrum)
        self.timer.start(100)

        # --- Данные ---
        self.center_freqs = np.arange(self.START_FREQ, self.END_FREQ, self.STEP)
        self.full_spectrum = np.zeros(len(self.center_freqs) * self.NUM_SAMPLES)
        self.full_freqs = np.zeros(len(self.center_freqs) * self.NUM_SAMPLES)

        # --- Max Hold ---
        self.max_hold_enabled = False
        self.max_hold_spectrum = np.full(len(self.center_freqs) * self.NUM_SAMPLES, -120.0)  # начальные значения
        self.max_hold_curve = self.graphWidget.plot(pen=pg.mkPen('c', style=pg.QtCore.Qt.DashLine))

    def get_window(self, window_type):
        if window_type == 'hann':
            return np.hanning(self.NUM_SAMPLES)
        elif window_type == 'hamming':
            return np.hamming(self.NUM_SAMPLES)
        elif window_type == 'blackman':
            return np.blackman(self.NUM_SAMPLES)
        else:
            return np.ones(self.NUM_SAMPLES)

    def init_sdr(self):
        self.rx.sample_rate = int(self.SAMPLE_RATE)
        self.rx.bandwidth = int(self.SPAN)
        self.rx.gain_mode = _bladerf.GainMode.Manual
        self.rx.gain = self.GAIN
        self.rx.enable = True

        self.sdr.sync_config(
            layout=_bladerf.ChannelLayout.RX_X1,
            fmt=_bladerf.Format.SC16_Q11,
            num_buffers=16,
            buffer_size=self.NUM_SAMPLES * 4,
            num_transfers=8,
            stream_timeout=3500
        )

    def init_ui(self):
        self.setWindowTitle(f"BladeRF 2.0 Spectrum Analyzer | Window: {self.WINDOW_TYPE}")
        self.setGeometry(100, 100, 1000, 600)

        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setLabel('left', 'Power (dBm)')
        self.graphWidget.setLabel('bottom', 'Frequency (MHz)')
        self.graphWidget.setYRange(-120, 0)
        self.graphWidget.showGrid(x=True, y=True)

        self.spectrum_curve = self.graphWidget.plot(pen='y')

        self.max_marker = pg.ScatterPlotItem(size=15, pen=pg.mkPen('r'), brush=pg.mkBrush('r'))
        self.graphWidget.addItem(self.max_marker)

        self.max_text = pg.TextItem(anchor=(0.5, 1.5), color='w', fill='k')
        self.graphWidget.addItem(self.max_text)

        # --- Кнопка включения/выключения Max Hold ---
        self.toggle_maxhold_btn = QPushButton("Max Hold: OFF")
        self.toggle_maxhold_btn.setCheckable(True)
        self.toggle_maxhold_btn.clicked.connect(self.toggle_max_hold)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.toggle_maxhold_btn)
        layout = QVBoxLayout()
        layout.addWidget(self.graphWidget)
        layout.addLayout(buttons_layout)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)


    def calculate_spectrum(self, samples):
        windowed_samples = samples * self.window
        spectrum = np.fft.fftshift(np.fft.fft(windowed_samples))
        power = (np.abs(spectrum) / (len(samples) * np.sqrt(self.window_correction))) ** 2
        power_db = 10 * np.log10(power + 1e-12)
        return power_db

    def get_calibration(self, freq):
        closest_freq = min(self.calibration_table.keys(), key=lambda f: abs(f - freq))
        return self.calibration_table[closest_freq]

    def toggle_max_hold(self):
        self.max_hold_enabled = not self.max_hold_enabled
        if self.max_hold_enabled:
            self.toggle_maxhold_btn.setText("Max Hold: ON")
        else:
            self.toggle_maxhold_btn.setText("Max Hold: OFF")
            self.max_hold_spectrum[:] = -120.0  # сброс
            self.max_hold_curve.setData([], [])

    def update_spectrum(self):
        scan_start = time.time()

        for i, freq in enumerate(self.center_freqs):
            cal = self.get_calibration(freq)
            freq_offset = cal["freq_offset"]
            power_offset = cal["power_offset"]

            self.rx.frequency = int(freq + freq_offset)
            time.sleep(0.0001)

            buf = bytearray(self.NUM_SAMPLES * 4)
            self.sdr.sync_rx(buf, self.NUM_SAMPLES)

            samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
            samples = samples.view(np.complex64) / (2 ** 11)

            power_db = self.calculate_spectrum(samples)
            power_db += power_offset

            start_idx = i * self.NUM_SAMPLES
            end_idx = (i + 1) * self.NUM_SAMPLES
            self.full_freqs[start_idx:end_idx] = np.fft.fftshift(
                np.fft.fftfreq(len(samples), d=1 / self.SAMPLE_RATE)) + freq
            self.full_spectrum[start_idx:end_idx] = power_db

        # Обновление Max Hold, если включено
        if self.max_hold_enabled:
            self.max_hold_spectrum = np.maximum(self.max_hold_spectrum, self.full_spectrum)
            self.max_hold_curve.setData(self.full_freqs / 1e6, self.max_hold_spectrum)

        # Поиск максимума
        max_idx = np.argmax(self.full_spectrum)
        max_freq = self.full_freqs[max_idx] / 1e6
        max_power = self.full_spectrum[max_idx]

        # Обновление графика
        self.spectrum_curve.setData(self.full_freqs / 1e6, self.full_spectrum)
        self.max_marker.setData([max_freq], [max_power])
        self.max_text.setText(f"Peak: {max_power:.1f} dBm @ {max_freq:.2f} MHz")
        self.max_text.setPos(max_freq, max_power)

        # Вывод времени сканирования
        scan_time = (time.time() - scan_start)
        print(f"Scan time: {scan_time:.2f} s | Peak: {max_power:.1f} dBm @ {max_freq:.2f} MHz")

    def closeEvent(self, event):
        self.rx.enable = False
        self.sdr.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpectrumAnalyzer()
    window.show()
    sys.exit(app.exec_())