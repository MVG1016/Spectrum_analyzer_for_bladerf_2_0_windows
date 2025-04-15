from bladerf import _bladerf
import numpy as np
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
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
        self.START_FREQ = 300e6
        self.END_FREQ = 6000e6
        self.SPAN = 30e6
        self.STEP = 20e6
        self.SAMPLE_RATE = 40e6
        self.NUM_SAMPLES = 8192  # Увеличено для лучшего разрешения
        self.GAIN = 18
        self.AVG_COUNT = 5
        self.WINDOW_TYPE = 'hann'  # Выбор окна: 'hann', 'hamming', 'blackman'

        # --- Калибровочные коэффициенты ---
        self.FREQ_OFFSET = 0  # Определяется экспериментально
        self.POWER_OFFSET = 0  # Определяется по эталонному сигналу

        # --- Инициализация SDR ---
        self.init_sdr()

        # --- Оконная функция ---
        self.window = self.get_window(self.WINDOW_TYPE)
        self.window_correction = np.mean(self.window ** 2)  # Поправка на потери в окне

        # --- GUI ---
        self.init_ui()

        # --- Таймер обновления ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_spectrum)
        self.timer.start(100)  # Обновление каждые 100 мс

        # --- Данные ---
        self.center_freqs = np.arange(self.START_FREQ, self.END_FREQ, self.STEP)
        self.full_spectrum = np.zeros(len(self.center_freqs) * self.NUM_SAMPLES)
        self.full_freqs = np.zeros(len(self.center_freqs) * self.NUM_SAMPLES)

    def get_window(self, window_type):
        """Возвращает выбранную оконную функцию"""
        if window_type == 'hann':
            return np.hanning(self.NUM_SAMPLES)
        elif window_type == 'hamming':
            return np.hamming(self.NUM_SAMPLES)
        elif window_type == 'blackman':
            return np.blackman(self.NUM_SAMPLES)
        else:
            return np.ones(self.NUM_SAMPLES)  # Прямоугольное окно

    def init_sdr(self):
        """Инициализация параметров SDR"""
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
        """Инициализация интерфейса"""
        self.setWindowTitle(f"BladeRF 2.0 Spectrum Analyzer | Window: {self.WINDOW_TYPE}")
        self.setGeometry(100, 100, 1000, 600)

        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setLabel('left', 'Power (dBm)')
        self.graphWidget.setLabel('bottom', 'Frequency (MHz)')
        self.graphWidget.setYRange(-120, 0)
        self.graphWidget.showGrid(x=True, y=True)

        self.spectrum_curve = self.graphWidget.plot(pen='y')

        # Маркер максимума
        self.max_marker = pg.ScatterPlotItem(size=15, pen=pg.mkPen('r'), brush=pg.mkBrush('r'))
        self.graphWidget.addItem(self.max_marker)

        # Текст с параметрами максимума
        self.max_text = pg.TextItem(anchor=(0.5, 1.5), color='w', fill='k')
        self.graphWidget.addItem(self.max_text)

        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.graphWidget)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def calculate_spectrum(self, samples):
        """Вычисление спектра с учетом оконной функции"""
        # Применение окна
        windowed_samples = samples * self.window

        # FFT
        spectrum = np.fft.fftshift(np.fft.fft(windowed_samples))

        # Расчет мощности с поправкой на окно
        power = (np.abs(spectrum) / (len(samples) * np.sqrt(self.window_correction))) ** 2
        power_db = 10 * np.log10(power + 1e-12) + self.POWER_OFFSET

        return power_db

    def update_spectrum(self):
        """Обновление спектра"""
        scan_start = time.time()

        for i, freq in enumerate(self.center_freqs):
            self.rx.frequency = int(freq + self.FREQ_OFFSET)
            time.sleep(0.01)

            buf = bytearray(self.NUM_SAMPLES * 4)
            self.sdr.sync_rx(buf, self.NUM_SAMPLES)

            samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
            samples = samples.view(np.complex64) / (2 ** 11)

            power_db = self.calculate_spectrum(samples)

            # Обновление данных
            start_idx = i * self.NUM_SAMPLES
            end_idx = (i + 1) * self.NUM_SAMPLES
            self.full_freqs[start_idx:end_idx] = np.fft.fftshift(
                np.fft.fftfreq(len(samples), d=1 / self.SAMPLE_RATE)) + freq
            self.full_spectrum[start_idx:end_idx] = power_db

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
        scan_time = (time.time() - scan_start) * 1
        print(f"Scan time: {scan_time:.1f} s | Peak: {max_power:.1f} dBm @ {max_freq:.2f} MHz")

    def closeEvent(self, event):
        """Остановка SDR при закрытии"""
        self.rx.enable = False
        self.sdr.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpectrumAnalyzer()
    window.show()
    sys.exit(app.exec_())