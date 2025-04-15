from bladerf import _bladerf
import numpy as np
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import pyqtgraph as pg

# --- Параметры ---
START_FREQ = 100e6
END_FREQ = 6000e6
SPAN = 15e6  # ширина захвата
STEP = 10e6  # шаг с перекрытием
SAMPLE_RATE = 40e6
NUM_SAMPLES = 4096
GAIN = 20
AVG_COUNT = 5  # Уменьшил для скорости обновления


class SpectrumAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()

        # Настройка окна
        self.setWindowTitle("BladeRF 2.0 — Реальный спектр с маркером")
        self.setGeometry(100, 100, 1000, 600)

        # Виджет для графика
        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setLabel('left', 'Амплитуда (dBFS)')
        self.graphWidget.setLabel('bottom', 'Частота (МГц)')
        self.graphWidget.setYRange(-100, 0)  # Диапазон dBFS
        self.graphWidget.showGrid(x=True, y=True)

        # Линия графика (жёлтая)
        self.spectrum_curve = self.graphWidget.plot(pen='y')

        # Маркер максимума (красная точка)
        self.max_marker = pg.ScatterPlotItem(size=15, pen=pg.mkPen('r'), brush=pg.mkBrush('r'))
        self.graphWidget.addItem(self.max_marker)

        # Текст с частотой и мощностью
        self.max_text = pg.TextItem(anchor=(0.5, 1.5), color='w', fill='k')
        self.graphWidget.addItem(self.max_text)

        # Настройка SDR
        self.sdr = _bladerf.BladeRF()
        self.rx = self.sdr.Channel(_bladerf.CHANNEL_RX(0))
        self.rx.sample_rate = int(SAMPLE_RATE)
        self.rx.bandwidth = int(SPAN)
        self.rx.gain_mode = _bladerf.GainMode.Manual
        self.rx.gain = GAIN
        self.rx.enable = True

        self.sdr.sync_config(
            layout=_bladerf.ChannelLayout.RX_X1,
            fmt=_bladerf.Format.SC16_Q11,
            num_buffers=16,
            buffer_size=NUM_SAMPLES * 4,
            num_transfers=8,
            stream_timeout=3500
        )

        # Таймер для обновления графика
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_spectrum)
        self.timer.start(100)  # Обновление каждые 100 мс

        # Центральный виджет
        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.graphWidget)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Частотные точки
        self.center_freqs = np.arange(START_FREQ, END_FREQ, STEP)
        self.full_spectrum = np.zeros(len(self.center_freqs) * NUM_SAMPLES)
        self.full_freqs = np.zeros(len(self.center_freqs) * NUM_SAMPLES)

    def update_spectrum(self):
        """Обновляет спектр и маркер максимума"""
        for i, freq in enumerate(self.center_freqs):
            self.rx.frequency = int(freq)
            time.sleep(0.01)

            # Получаем данные
            buf = bytearray(NUM_SAMPLES * 4)
            self.sdr.sync_rx(buf, NUM_SAMPLES)

            samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
            samples = samples.view(np.complex64) / (2 ** 11)

            # FFT
            windowed = samples * np.hanning(len(samples))
            spectrum = np.fft.fftshift(np.fft.fft(windowed))
            power_db = 20 * np.log10(np.abs(spectrum) / len(samples) + 1e-12)

            # Обновляем частоты и спектр
            start_idx = i * NUM_SAMPLES
            end_idx = (i + 1) * NUM_SAMPLES
            self.full_freqs[start_idx:end_idx] = np.fft.fftshift(np.fft.fftfreq(len(samples), d=1 / SAMPLE_RATE)) + freq
            self.full_spectrum[start_idx:end_idx] = power_db

        # Находим максимум
        max_idx = np.argmax(self.full_spectrum)
        max_freq = self.full_freqs[max_idx] / 1e6  # в МГц
        max_power = self.full_spectrum[max_idx]

        # Обновляем график
        self.spectrum_curve.setData(self.full_freqs / 1e6, self.full_spectrum)

        # Обновляем маркер
        self.max_marker.setData([max_freq], [max_power])

        # Обновляем текст
        self.max_text.setText(f"Макс: {max_power:.1f} дБ @ {max_freq:.2f} МГц", color='w')
        self.max_text.setPos(max_freq, max_power)

    def closeEvent(self, event):
        """При закрытии окна останавливаем SDR"""
        self.rx.enable = False
        self.sdr.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication([])
    window = SpectrumAnalyzer()
    window.show()
    app.exec_()