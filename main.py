from bladerf import _bladerf
import numpy as np
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QComboBox, QTabWidget, QLineEdit, QLabel
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import QLocale
import pyqtgraph as pg
import sys
from PyQt5.QtGui import QDoubleValidator
from threading import Thread
import sys
import os
from datetime import datetime


class SpectrumAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- Конфигурация SDR ---
        self.sdr = _bladerf.BladeRF()
        self.rx = None  # Инициализируется позже в switch_rx_channel
        self.tx = None  # Инициализируется позже в switch_tx_channel
        self.sweep_active = False
        self.sweep_thread = None


        # --- Параметры ---
        self.START_FREQ = 5000e6
        self.END_FREQ = 5700e6
        self.SPAN = 20e6
        self.STEP = 10e6
        self.SAMPLE_RATE = 40e6
        self.NUM_SAMPLES = 8192
        self.GAIN = 18
        self.WINDOW_TYPE = 'hann'

        # --- Калибровочная таблица ---
        self.calibration_table = {
            100e6:  {"freq_offset": 0e6, "power_offset": 0},
            500e6:  {"freq_offset": 0e6, "power_offset": 0},
            1000e6: {"freq_offset": 0e6, "power_offset": 0},
            2000e6: {"freq_offset": 0e6, "power_offset": 0},
            2400e6: {"freq_offset": 0e6, "power_offset": 0},
            3000e6: {"freq_offset": 0e6, "power_offset": 0},
            4000e6: {"freq_offset": 0e6, "power_offset": 0},
            5000e6: {"freq_offset": 0e6, "power_offset": 0},
            5800e6: {"freq_offset": 0e6, "power_offset": 0},
        }

        # --- Оконная функция ---
        self.window = self.get_window(self.WINDOW_TYPE)
        self.window_correction = np.mean(self.window ** 2)

        # --- GUI ---
        self.init_ui()
        self.switch_rx_channel(0)  # RX1 по умолчанию
        self.switch_tx_channel(0)  # TX1 по умолчанию

        # --- Таймер ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_spectrum)

        # --- Данные ---
        self.center_freqs = np.arange(self.START_FREQ, self.END_FREQ, self.STEP)
        self.full_spectrum = np.zeros(len(self.center_freqs) * self.NUM_SAMPLES)
        self.full_freqs = np.zeros(len(self.center_freqs) * self.NUM_SAMPLES)

        # --- Max Hold ---
        self.max_hold_enabled = False
        self.max_hold_spectrum = np.full(len(self.center_freqs) * self.NUM_SAMPLES, -120.0)
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

    def switch_rx_channel(self, index):
        if self.rx is not None:
            self.rx.enable = False
        self.rx = self.sdr.Channel(_bladerf.CHANNEL_RX(index))
        self.init_rx_sdr()

    def switch_tx_channel(self, index):
        if self.tx is not None:
            self.tx.enable = False
        self.tx = self.sdr.Channel(_bladerf.CHANNEL_TX(index))
        self.init_tx_sdr()

    def init_rx_sdr(self):
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

    def init_tx_sdr(self):
        self.tx.sample_rate = int(self.SAMPLE_RATE)
        self.tx.bandwidth = int(self.SPAN)
        self.tx.gain_mode = _bladerf.GainMode.Manual
        self.tx.gain = self.GAIN
        self.tx.enable = True

        self.sdr.sync_config(
            layout=_bladerf.ChannelLayout.TX_X1,
            fmt=_bladerf.Format.SC16_Q11,
            num_buffers=16,
            buffer_size=self.NUM_SAMPLES * 4,
            num_transfers=8,
            stream_timeout=3500
        )

    def init_ui(self):
        self.setWindowTitle(f"BladeRF 2.0 Spectrum Analyzer and Signal GAenerator")
        self.setGeometry(100, 100, 1000, 600)

        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        self.rx_tab = QWidget()
        self.tx_tab = QWidget()
        self.tab_widget.addTab(self.rx_tab, "Receive Channel")
        self.tab_widget.addTab(self.tx_tab, "Transmit Channel")

        self.init_rx_ui()
        self.init_tx_ui()

    def init_rx_ui(self):
        layout = QVBoxLayout()

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

        self.toggle_maxhold_btn = QPushButton("Max Hold: OFF")
        self.toggle_maxhold_btn.setCheckable(True)
        self.toggle_maxhold_btn.clicked.connect(self.toggle_max_hold)

        self.scan_btn = QPushButton("Start Scan")
        self.scan_btn.setCheckable(True)
        self.scan_btn.clicked.connect(self.toggle_scan)

        self.rx_selector = QComboBox()
        self.rx_selector.addItems(["RX1", "RX2"])
        self.rx_selector.currentIndexChanged.connect(self.switch_rx_channel)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.toggle_maxhold_btn)
        buttons_layout.addWidget(self.scan_btn)
        buttons_layout.addWidget(self.rx_selector)

        params_layout = QHBoxLayout()

        self.start_freq_input = QLineEdit(str(self.START_FREQ / 1e6))
        self.start_freq_input.setValidator(QDoubleValidator(1, 6000, 2))
        self.start_freq_input.setFixedWidth(80)

        self.end_freq_input = QLineEdit(str(self.END_FREQ / 1e6))
        self.end_freq_input.setValidator(QDoubleValidator(1, 6000, 2))
        self.end_freq_input.setFixedWidth(80)

        self.step_input = QLineEdit(str(self.STEP / 1e6))
        self.step_input.setValidator(QDoubleValidator(0.1, 100, 2))
        self.step_input.setFixedWidth(80)

        self.span_input = QLineEdit(str(self.SPAN / 1e6))
        self.span_input.setValidator(QDoubleValidator(1, 100, 2))
        self.span_input.setFixedWidth(80)

        self.sample_rate_input = QLineEdit(str(self.SAMPLE_RATE / 1e6))
        self.sample_rate_input.setValidator(QDoubleValidator(1, 100, 2))
        self.sample_rate_input.setFixedWidth(80)

        self.num_samples_input = QLineEdit(str(self.NUM_SAMPLES))
        self.num_samples_input.setValidator(QDoubleValidator(128, 1_000_000, 0))
        self.num_samples_input.setFixedWidth(80)

        self.gain_input = QLineEdit(str(self.GAIN))
        self.gain_input.setValidator(QDoubleValidator(0, 60, 0))
        self.gain_input.setFixedWidth(80)

        self.apply_rx_settings_btn = QPushButton("Apply RX Settings")
        self.apply_rx_settings_btn.clicked.connect(self.apply_rx_settings)

        params_layout.addWidget(QLabel("Start (MHz):"))
        params_layout.addWidget(self.start_freq_input)
        params_layout.addWidget(QLabel("Stop (MHz):"))
        params_layout.addWidget(self.end_freq_input)
        params_layout.addWidget(QLabel("Step (MHz):"))
        params_layout.addWidget(self.step_input)
        params_layout.addWidget(QLabel("Span (MHz):"))
        params_layout.addWidget(self.span_input)
        params_layout.addWidget(QLabel("Rate (MHz):"))
        params_layout.addWidget(self.sample_rate_input)
        params_layout.addWidget(QLabel("Samples:"))
        params_layout.addWidget(self.num_samples_input)
        params_layout.addWidget(QLabel("Gain:"))
        params_layout.addWidget(self.gain_input)
        params_layout.addWidget(self.apply_rx_settings_btn)

        layout.addLayout(params_layout)


        layout.addWidget(self.graphWidget)
        layout.addLayout(buttons_layout)
        self.rx_tab.setLayout(layout)

    def init_tx_ui(self):
        layout = QVBoxLayout()

        # Кнопка для начала передачи
        self.start_tx_btn = QPushButton("Start Transmission")
        self.start_tx_btn.setCheckable(True)
        self.start_tx_btn.clicked.connect(self.toggle_transmission)
        self.start_tx_btn.setFixedWidth(300)

        # Поле для ввода частоты
        self.tx_freq_label = QLabel("Frequency: 5000 MHz")
        self.tx_freq_input = QLineEdit("5000")
        self.tx_freq_input.setValidator(QDoubleValidator(0, 6000, 2))
        self.tx_freq_input.setFixedWidth(100)

        # Поле для ввода мощности
        self.tx_power_label = QLabel("Power: 10 dB")
        self.tx_power_input = QLineEdit("10")
        self.tx_power_input.setValidator(QDoubleValidator(0, 30, 1))
        self.tx_power_input.setFixedWidth(100)


        # Выпадающий список для выбора канала
        self.tx_selector = QComboBox()
        self.tx_selector.addItems(["TX1", "TX2"])
        self.tx_selector.currentIndexChanged.connect(self.switch_tx_channel)
        self.tx_selector.setFixedWidth(100)

        # Sweep mode controls
        self.sweep_enable_btn = QPushButton("Start Sweep Mode")
        self.sweep_enable_btn.setCheckable(True)
        self.sweep_enable_btn.clicked.connect(self.toggle_sweep_mode)
        self.sweep_enable_btn.setFixedWidth(300)



        self.sweep_start_freq = QLineEdit("5000")
        self.sweep_start_freq.setValidator(QDoubleValidator(0, 6000, 2))
        self.sweep_start_freq.setFixedWidth(100)

        self.sweep_stop_freq = QLineEdit("5700")
        self.sweep_stop_freq.setValidator(QDoubleValidator(0, 6000, 2))
        self.sweep_stop_freq.setFixedWidth(100)


        self.sweep_step_freq = QLineEdit("10")
        self.sweep_step_freq.setValidator(QDoubleValidator(0.01, 1000, 2))
        self.sweep_step_freq.setFixedWidth(100)

        self.sweep_delay = QLineEdit("100")
        self.sweep_delay.setValidator(QDoubleValidator(1, 10000, 0))
        self.sweep_delay.setFixedWidth(100)


        layout.addWidget(QLabel("Sweep Start Freq (MHz):"))
        layout.addWidget(self.sweep_start_freq)
        layout.addWidget(QLabel("Sweep Stop Freq (MHz):"))
        layout.addWidget(self.sweep_stop_freq)
        layout.addWidget(QLabel("Sweep Step (MHz):"))
        layout.addWidget(self.sweep_step_freq)
        layout.addWidget(QLabel("Delay per step (ms):"))
        layout.addWidget(self.sweep_delay)
        layout.addWidget(self.sweep_enable_btn)


        # Добавление элементов на интерфейс
        layout.addWidget(self.start_tx_btn)
        layout.addWidget(self.tx_freq_label)
        layout.addWidget(self.tx_freq_input)
        layout.addWidget(self.tx_power_label)
        layout.addWidget(self.tx_power_input)
        layout.addWidget(self.tx_selector)

        self.tx_tab.setLayout(layout)

    def apply_rx_settings(self):
        try:
            #  Останавливаем таймер обновления спектра, если он работает
            if self.timer.isActive():
                self.timer.stop()
                print("Spectrum update stopped.")

            #  Отключаем приём перед изменением параметров
            if self.rx is not None:
                self.rx.enable = False
                time.sleep(0.2)

            self.START_FREQ = float(self.start_freq_input.text()) * 1e6
            self.END_FREQ = float(self.end_freq_input.text()) * 1e6
            self.STEP = float(self.step_input.text()) * 1e6
            self.SPAN = float(self.span_input.text()) * 1e6
            self.SAMPLE_RATE = float(self.sample_rate_input.text()) * 1e6
            self.NUM_SAMPLES = int(float(self.num_samples_input.text()))
            self.GAIN = int(float(self.gain_input.text()))

            self.window = self.get_window(self.WINDOW_TYPE)
            self.window_correction = np.mean(self.window ** 2)

            self.center_freqs = np.arange(self.START_FREQ, self.END_FREQ, self.STEP)
            self.full_spectrum = np.zeros(len(self.center_freqs) * self.NUM_SAMPLES)
            self.full_freqs = np.zeros(len(self.center_freqs) * self.NUM_SAMPLES)
            self.max_hold_spectrum = np.full(len(self.center_freqs) * self.NUM_SAMPLES, -120.0)

            self.init_rx_sdr()
            print("RX parameters updated.")
        except Exception as e:
            print(f"Failed to apply RX settings: {e}")




    def set_tx_frequency(self):
        try:
            freq = float(self.tx_freq_input.text()) * 1e6  # преобразуем в Hz
            self.tx.frequency = int(freq)
            print(f"Set TX Frequency: {freq / 1e6} MHz")
        except ValueError:
            print("Invalid frequency input")

    def set_tx_power(self):
        try:
            power = float(self.tx_power_input.text())  # дБ
            self.tx.gain = power
            print(f"Set TX Power: {power} dB")
        except ValueError:
            print("Invalid power input")

    def start_transmission(self):
        try:
            freq = float(self.tx_freq_input.text()) * 1e6
            power = float(self.tx_power_input.text())

            self.tx.frequency = int(freq)
            self.tx.gain = int(round(power))
            self.tx.enable = True

            # Пример генерации синуса на 1 МГц
            t = np.arange(self.NUM_SAMPLES) / self.SAMPLE_RATE
            tx_signal = 0.7 * np.exp(2j * np.pi * 1e6 * t)  # комплексный синус

            # Преобразование в формат SC16 Q11
            iq = np.empty(2 * len(tx_signal), dtype=np.int16)
            iq[0::2] = np.clip(np.real(tx_signal) * (2 ** 11), -2048, 2047).astype(np.int16)
            iq[1::2] = np.clip(np.imag(tx_signal) * (2 ** 11), -2048, 2047).astype(np.int16)

            # Передача одного буфера
            self.sdr.sync_tx(iq.tobytes(), len(tx_signal))

            print(f"Started transmission at {freq / 1e6} MHz with {self.tx.gain} dB power")

            self.tx_freq_label.setText(f"TX Frequency: {freq / 1e6} MHz")
            self.tx_power_label.setText(f"TX Power: {self.tx.gain} dB")

        except ValueError:
            print("Invalid inputs for frequency or power.")
            self.tx_freq_label.setText("Invalid frequency or power")


    def toggle_sweep_mode(self):
        if self.sweep_enable_btn.isChecked():
            self.sweep_enable_btn.setText("Stop Sweep Mode")
            self.sweep_active = True
            self.sweep_thread = Thread(target=self.run_sweep_transmission)
            self.sweep_thread.start()
        else:
            self.sweep_active = False
            self.sweep_enable_btn.setText("Start Sweep Mode")
            if self.tx:
                self.tx.enable = False
            print("Sweep transmission stopped.")


    def run_sweep_transmission(self):
        try:
            self.tx.sample_rate = int(self.SAMPLE_RATE)
            self.tx.bandwidth = int(self.SPAN)
            self.tx.gain_mode = _bladerf.GainMode.Manual
            self.tx.gain = self.GAIN
            self.tx.enable = True

            self.sdr.sync_config(
                layout=_bladerf.ChannelLayout.TX_X1,
                fmt=_bladerf.Format.SC16_Q11,
                num_buffers=16,
                buffer_size=self.NUM_SAMPLES * 4,
                num_transfers=8,
                stream_timeout=3500
            )


            start_freq = float(self.sweep_start_freq.text()) * 1e6
            stop_freq = float(self.sweep_stop_freq.text()) * 1e6
            step_freq = float(self.sweep_step_freq.text()) * 1e6
            delay_ms = int(self.sweep_delay.text())

            t = np.arange(self.NUM_SAMPLES) / self.SAMPLE_RATE
            tx_signal = 0.7 * np.exp(2j * np.pi * 1e6 * t)

            iq = np.empty(2 * len(tx_signal), dtype=np.int16)
            iq[0::2] = np.clip(np.real(tx_signal) * (2 ** 11), -2048, 2047).astype(np.int16)
            iq[1::2] = np.clip(np.imag(tx_signal) * (2 ** 11), -2048, 2047).astype(np.int16)

            while self.sweep_active:
                for freq in np.arange(start_freq, stop_freq + step_freq, step_freq):
                    if not self.sweep_active:
                        break

                    power = float(self.tx_power_input.text())  # Считываем мощность из поля ввода
                    self.tx.gain = int(round(power))  # Применяем мощность к передатчику
                    self.tx.frequency = int(freq)
                    self.tx.enable = True
                    self.sdr.sync_tx(iq.tobytes(), len(tx_signal))
                    print(f"Sweep TX @ {freq/1e6:.2f} MHz")
                    self.tx_power_label.setText(f"TX Power: {self.tx.gain} dB")
                    time.sleep(delay_ms / 1000.0)

            if self.tx:
                self.tx.enable = False

        except Exception as e:
            print(f"Error in sweep mode: {e}")


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
            self.max_hold_spectrum[:] = -120.0
            self.max_hold_curve.setData([], [])

    def toggle_scan(self):
        if self.scan_btn.isChecked():
            self.scan_btn.setText("Stop Scan")
            self.timer.start(100)
        else:
            self.scan_btn.setText("Start Scan")
            self.timer.stop()

    def toggle_transmission(self):
        if self.start_tx_btn.isChecked():
            try:
                freq = float(self.tx_freq_input.text()) * 1e6
                power = float(self.tx_power_input.text())

                self.tx.frequency = int(freq)
                self.tx.gain = int(round(power))
                self.tx.enable = True

                self.tx_freq_label.setText(f"TX Frequency: {freq / 1e6} MHz")
                self.tx_power_label.setText(f"TX Power: {self.tx.gain} dB")
                self.start_tx_btn.setText("Stop Transmission")

                print(f"Started transmission at {freq / 1e6} MHz with {self.tx.gain} dB power")

            except ValueError:
                print("Invalid frequency or power input")
                self.tx_freq_label.setText("Invalid frequency or power")
        else:
            self.tx.enable = False
            self.start_tx_btn.setText("Start Transmission")
            print("Transmission stopped")

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

        if self.max_hold_enabled:
            self.max_hold_spectrum = np.maximum(self.max_hold_spectrum, self.full_spectrum)
            self.max_hold_curve.setData(self.full_freqs / 1e6, self.max_hold_spectrum)

        max_idx = np.argmax(self.full_spectrum)
        max_freq = self.full_freqs[max_idx] / 1e6
        max_power = self.full_spectrum[max_idx]

        self.spectrum_curve.setData(self.full_freqs / 1e6, self.full_spectrum)
        self.max_marker.setData([max_freq], [max_power])
        self.max_text.setText(f"Peak: {max_power:.1f} dBm @ {max_freq:.2f} MHz")
        self.max_text.setPos(max_freq, max_power)

        scan_time = (time.time() - scan_start)
        print(f"Scan time: {scan_time:.2f} s | Peak: {max_power:.1f} dBm @ {max_freq:.2f} MHz")

    def closeEvent(self, event):
        if self.rx is not None:
            self.rx.enable = False
        if self.tx is not None:
            self.tx.enable = False
        #self.sweep_active = False
        self.sdr.close()
        event.accept()


if __name__ == "__main__":
    QLocale.setDefault(QLocale("C"))
    # Создаем имя лог-файла по текущей дате и времени
    log_filename = datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.txt")
    log_path = os.path.join(os.path.dirname(__file__), log_filename)



    # Класс для дублирования вывода в консоль и файл
    class Logger:
        def __init__(self, filepath):
            self.terminal = sys.__stdout__  # Используем sys.__stdout__ напрямую (всегда существует)
            self.log = open(filepath, "w", encoding="utf-8")

        def write(self, message):
            try:
                if self.terminal:
                    self.terminal.write(message)
            except Exception:
                pass  # На всякий случай — не падать даже при сбоях вывода

            try:
                self.log.write(message)
            except Exception:
                pass

        def flush(self):
            try:
                if self.terminal:
                    self.terminal.flush()
            except Exception:
                pass

            try:
                self.log.flush()
            except Exception:
                pass


    # Перехватываем stdout и stderr
    logger = Logger(log_path)
    sys.stdout = logger
    sys.stderr = logger
    print(f"Лог-файл запущен: {log_path}")

    app = QApplication(sys.argv)
    window = SpectrumAnalyzer()
    window.show()
    sys.exit(app.exec_())