"""
Wideband Spectrum Analyzer - BladeRF 2.0 + USRP B205mini
Supports device selection with unified interface
"""

import numpy as np
import time
import sys
import os
from abc import ABC, abstractmethod
from datetime import datetime
from threading import Lock
from dataclasses import dataclass

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                             QPushButton, QHBoxLayout, QComboBox, QFormLayout,
                             QLineEdit, QLabel, QMessageBox, QSpinBox, QFrame,
                             QSlider, QGroupBox, QFileDialog, QScrollArea)
from PyQt5.QtCore import QTimer, QLocale, QThread, QRectF, Qt
from PyQt5.QtGui import QIcon
import pyqtgraph as pg


# ============================================================================
# Logging System
# ============================================================================

class Logger:
    """Logger that writes to both console and file"""

    def __init__(self, filepath: str):
        self.terminal = sys.__stdout__
        self.log = open(filepath, "w", encoding="utf-8", buffering=1)

    def write(self, message: str):
        if self.terminal:
            try:
                self.terminal.write(message)
            except Exception:
                pass
        try:
            self.log.write(message)
            self.log.flush()
        except Exception:
            pass

    def flush(self):
        if self.terminal:
            try:
                self.terminal.flush()
            except Exception:
                pass
        try:
            self.log.flush()
        except Exception:
            pass


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SDRConfig:
    """SDR configuration parameters"""
    start_freq: float = 70e6
    stop_freq: float = 6000e6
    step: float = 0.6e6
    sample_rate: float = 0.6e6
    num_samples: int = 4096
    gain: int = 30
    waterfall_lines: int = 30
    SYNC_NUM_BUFFERS: int = 16
    SYNC_NUM_TRANSFERS: int = 8
    SYNC_STREAM_TIMEOUT: int = 3500
    BUFFER_SIZE_MULTIPLIER: int = 4


CALIBRATION_TABLE = {
    400e6:  -77,
    800e6:  -77,
    1500e6: -77,
    2400e6: -77,
    3000e6: -77,
    5000e6: -77,
}


def get_calibration(freq_hz: float) -> float:
    freqs = np.array(list(CALIBRATION_TABLE.keys()))
    gains = np.array(list(CALIBRATION_TABLE.values()))
    return np.interp(freq_hz, freqs, gains)


# ============================================================================
# Abstract SDR Backend
# ============================================================================

class SDRBackend(ABC):
    """Abstract base class for SDR hardware backends"""

    @property
    def num_rx_channels(self) -> int:
        return 1

    @property
    def num_tx_channels(self) -> int:
        return 1

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def configure_rx(self, channel: int, freq: float, sample_rate: float,
                     gain: int, num_samples: int, config: SDRConfig) -> None: ...

    @abstractmethod
    def set_rx_freq(self, freq: float) -> None: ...

    @abstractmethod
    def set_rx_gain(self, gain: int) -> None: ...

    @abstractmethod
    def read_samples(self, num_samples: int) -> np.ndarray: ...

    @abstractmethod
    def flush_rx_buffer(self, num_samples: int, flush_count: int) -> None: ...

    @abstractmethod
    def configure_tx(self, channel: int, freq: float, sample_rate: float,
                     gain: int, num_samples: int, config: SDRConfig) -> None: ...

    @abstractmethod
    def set_tx_freq(self, freq: float) -> None: ...

    @abstractmethod
    def write_samples(self, iq_bytes: bytes, num_samples: int) -> None: ...

    @abstractmethod
    def disable_tx(self) -> None: ...

    @abstractmethod
    def reconfigure_rx_after_tx(self, num_samples: int, config: SDRConfig) -> None: ...

    @abstractmethod
    def get_actual_sample_rate(self) -> float: ...


# ============================================================================
# BladeRF Backend
# ============================================================================

class BladeRFBackend(SDRBackend):
    """BladeRF 2.0 backend"""

    def __init__(self):
        from bladerf import _bladerf as bf
        self._bf = bf
        self.sdr = None
        self.rx_channel = None
        self.tx_channel = None
        self._actual_sample_rate = 0.6e6

    @property
    def num_rx_channels(self) -> int:
        return 2

    @property
    def num_tx_channels(self) -> int:
        return 2

    def open(self) -> None:
        self.sdr = self._bf.BladeRF()
        print(f"BladeRF opened: {self.sdr.device_speed}")

    def close(self) -> None:
        if self.sdr:
            try:
                self.sdr.close()
            except Exception as e:
                print(f"BladeRF close error: {e}")

    def _sync_config(self, layout, num_samples: int, config: SDRConfig):
        self.sdr.sync_config(
            layout=layout,
            fmt=self._bf.Format.SC16_Q11,
            num_buffers=config.SYNC_NUM_BUFFERS,
            buffer_size=num_samples * config.BUFFER_SIZE_MULTIPLIER,
            num_transfers=config.SYNC_NUM_TRANSFERS,
            stream_timeout=config.SYNC_STREAM_TIMEOUT,
        )

    def configure_rx(self, channel: int, freq: float, sample_rate: float,
                     gain: int, num_samples: int, config: SDRConfig) -> None:
        if self.rx_channel is not None:
            try:
                self.rx_channel.enable = False
            except Exception:
                pass
            time.sleep(0.2)

        try:
            self.rx_channel = self.sdr.Channel(self._bf.CHANNEL_RX(channel))
        except Exception:
            ch = self._bf.CHANNEL_RX1 if channel == 0 else self._bf.CHANNEL_RX2
            self.rx_channel = self.sdr.Channel(ch)

        self.rx_channel.sample_rate = int(sample_rate)
        self.rx_channel.bandwidth = int(sample_rate)
        self.rx_channel.gain_mode = self._bf.GainMode.Manual
        self.rx_channel.gain = gain
        self.rx_channel.frequency = int(freq)

        self._sync_config(self._bf.ChannelLayout.RX_X1, num_samples, config)
        self.rx_channel.enable = True
        self._actual_sample_rate = float(self.rx_channel.sample_rate)
        time.sleep(0.1)

    def set_rx_freq(self, freq: float) -> None:
        self.rx_channel.frequency = int(freq)

    def set_rx_gain(self, gain: int) -> None:
        self.rx_channel.gain = gain

    def read_samples(self, num_samples: int) -> np.ndarray:
        buf = bytearray(num_samples * 4)
        self.sdr.sync_rx(buf, num_samples)
        samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
        samples = samples.view(np.complex64)
        samples /= 2048.0
        return samples

    def flush_rx_buffer(self, num_samples: int, flush_count: int) -> None:
        flush_buf = bytearray(num_samples * 4)
        for _ in range(flush_count):
            try:
                self.sdr.sync_rx(flush_buf, num_samples)
            except Exception:
                break

    def configure_tx(self, channel: int, freq: float, sample_rate: float,
                     gain: int, num_samples: int, config: SDRConfig) -> None:
        if self.tx_channel is not None:
            try:
                self.tx_channel.enable = False
            except Exception:
                pass
            time.sleep(0.2)

        self.tx_channel = self.sdr.Channel(self._bf.CHANNEL_TX(channel))
        self.tx_channel.sample_rate = int(sample_rate)
        self.tx_channel.bandwidth = int(sample_rate)
        self.tx_channel.gain_mode = self._bf.GainMode.Manual
        self.tx_channel.gain = gain
        self.tx_channel.frequency = int(freq)

        self._sync_config(self._bf.ChannelLayout.TX_X1, num_samples, config)
        self.tx_channel.enable = True
        time.sleep(0.05)

    def set_tx_freq(self, freq: float) -> None:
        if self.tx_channel:
            self.tx_channel.frequency = int(freq)

    def write_samples(self, iq_bytes: bytes, num_samples: int) -> None:
        self.sdr.sync_tx(iq_bytes, num_samples)

    def disable_tx(self) -> None:
        if self.tx_channel:
            try:
                self.tx_channel.enable = False
            except Exception:
                pass

    def reconfigure_rx_after_tx(self, num_samples: int, config: SDRConfig) -> None:
        time.sleep(0.1)
        self._sync_config(self._bf.ChannelLayout.RX_X1, num_samples, config)
        if self.rx_channel:
            self.rx_channel.enable = True

    def get_actual_sample_rate(self) -> float:
        return self._actual_sample_rate

    def set_sample_rate(self, sample_rate: float, num_samples: int, config: SDRConfig):
        self.rx_channel.enable = False
        self.rx_channel.sample_rate = int(sample_rate)
        self.rx_channel.bandwidth = int(sample_rate)
        self._sync_config(self._bf.ChannelLayout.RX_X1, num_samples, config)
        self.rx_channel.enable = True
        self._actual_sample_rate = float(self.rx_channel.sample_rate)


# ============================================================================
# USRP Backend
# ============================================================================

class USRPBackend(SDRBackend):
    """USRP B205mini backend (single RX + single TX).

    Ключевые принципы параллельной работы RX+TX:
    - RX работает в режиме start_cont непрерывно
    - При configure_tx: останавливаем RX, настраиваем TX, пересоздаём оба стримера
      и сразу запускаем оба — это единственный надёжный способ на B205mini
      с общим PLL
    - При disable_tx: завершаем TX burst, пересоздаём RX стример заново
      (старый стример после совместной работы с TX содержит мусор в буфере)
    - flush после disable_tx обязателен чтобы убрать накопившийся мусор
    """

    def __init__(self):
        self._uhd = None
        self.usrp = None
        self.rx_streamer = None
        self.tx_streamer = None
        self._sample_rate = 0.6e6
        self._rx_channel = 0
        self._tx_channel = 0
        self._tx_md = None
        self._tx_first_send = True
        self._tx_active = False  # флаг: TX стример сейчас работает

    @property
    def num_rx_channels(self) -> int:
        return 1

    @property
    def num_tx_channels(self) -> int:
        return 1

    def open(self) -> None:
        import uhd
        self._uhd = uhd
        self.usrp = uhd.usrp.MultiUSRP()
        print(f"USRP opened: {self.usrp.get_mboard_name()}")

    def close(self) -> None:
        self._teardown_rx_streamer()
        self._teardown_tx_streamer()
        self.usrp = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _teardown_rx_streamer(self):
        """Корректно остановить и уничтожить RX стример."""
        if self.rx_streamer is not None:
            try:
                cmd = self._uhd.types.StreamCMD(
                    self._uhd.types.StreamMode.stop_cont)
                self.rx_streamer.issue_stream_cmd(cmd)
                time.sleep(0.02)
            except Exception:
                pass
            self.rx_streamer = None

    def _teardown_tx_streamer(self):
        """Отправить end_of_burst и уничтожить TX стример."""
        if self.tx_streamer is not None:
            try:
                md = self._uhd.types.TXMetadata()
                md.end_of_burst = True
                md.start_of_burst = False
                zeros = np.zeros(256, dtype=np.complex64)
                self.tx_streamer.send(zeros, md)
                time.sleep(0.02)
            except Exception:
                pass
            self.tx_streamer = None
        self._tx_first_send = True
        self._tx_active = False

    def _build_and_start_rx(self, channel: int):
        """Создать RX стример и запустить непрерывный поток."""
        st_args = self._uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [channel]
        self.rx_streamer = self.usrp.get_rx_stream(st_args)
        cmd = self._uhd.types.StreamCMD(
            self._uhd.types.StreamMode.start_cont)
        cmd.stream_now = True
        self.rx_streamer.issue_stream_cmd(cmd)

    def _build_tx(self, channel: int):
        """Создать TX стример и подготовить метаданные."""
        tx_args = self._uhd.usrp.StreamArgs("fc32", "fc32")
        tx_args.channels = [channel]
        self.tx_streamer = self.usrp.get_tx_stream(tx_args)
        self._tx_md = self._uhd.types.TXMetadata()
        self._tx_md.start_of_burst = True
        self._tx_md.end_of_burst = False
        self._tx_md.has_time_spec = False
        self._tx_first_send = True
        self._tx_active = True

    def _drain_rx(self, num_samples: int, count: int = 5):
        """Вычитать несколько буферов чтобы сбросить накопившийся мусор."""
        if self.rx_streamer is None:
            return
        md = self._uhd.types.RXMetadata()
        buf = np.zeros(num_samples, dtype=np.complex64)
        for _ in range(count):
            try:
                self.rx_streamer.recv(buf, md, 0.1)
            except Exception:
                break

    # ------------------------------------------------------------------
    # SDRBackend interface
    # ------------------------------------------------------------------

    def configure_rx(self, channel: int, freq: float, sample_rate: float,
                     gain: int, num_samples: int, config: SDRConfig) -> None:
        """Настроить RX. TX стример при этом не затрагивается."""
        self._teardown_rx_streamer()
        self._rx_channel = channel
        self._sample_rate = sample_rate

        self.usrp.set_rx_rate(sample_rate, channel)
        self.usrp.set_rx_freq(self._uhd.types.TuneRequest(freq), channel)
        self.usrp.set_rx_gain(gain, channel)
        self.usrp.set_rx_antenna("RX2", channel)

        self._build_and_start_rx(channel)
        time.sleep(0.05)

    def set_rx_freq(self, freq: float) -> None:
        self.usrp.set_rx_freq(
            self._uhd.types.TuneRequest(freq), self._rx_channel)

    def set_rx_gain(self, gain: int) -> None:
        self.usrp.set_rx_gain(gain, self._rx_channel)

    def read_samples(self, num_samples: int) -> np.ndarray:
        """Читать ровно num_samples из непрерывного RX стримера."""
        buf = np.zeros(num_samples, dtype=np.complex64)
        md = self._uhd.types.RXMetadata()
        received = 0
        while received < num_samples:
            n = self.rx_streamer.recv(buf[received:], md, 0.5)
            if md.error_code not in (
                self._uhd.types.RXMetadataErrorCode.none,
                self._uhd.types.RXMetadataErrorCode.overflow,
            ):
                print(f"USRP RX error: {md.strerror()}")
                break
            received += n
        return buf

    def flush_rx_buffer(self, num_samples: int, flush_count: int) -> None:
        """Сдренировать устаревшие отсчёты после перестройки частоты.

        Не останавливаем стример — только вычитываем несколько буферов.
        """
        self._drain_rx(num_samples, count=3)

    def configure_tx(self, channel: int, freq: float, sample_rate: float,
                     gain: int, num_samples: int, config: SDRConfig) -> None:
        """Настроить TX рядом с работающим RX на B205mini.

        На B205mini PLL общий для RX и TX. Единственный надёжный способ
        получить одновременную работу — остановить RX, настроить TX,
        затем пересоздать оба стримера и запустить их вместе.
        Это добавляет ~150 мс паузы в RX, но гарантирует корректную работу.
        """
        # 1. Остановить текущий RX стример
        self._teardown_rx_streamer()
        # 2. Завершить предыдущий TX если был
        self._teardown_tx_streamer()

        # 3. Настроить TX через USRP API
        self._tx_channel = channel
        self.usrp.set_tx_rate(sample_rate, channel)
        self.usrp.set_tx_freq(self._uhd.types.TuneRequest(freq), channel)
        self.usrp.set_tx_gain(gain, channel)
        self.usrp.set_tx_antenna("TX/RX", channel)

        # 4. Создать TX стример
        self._build_tx(channel)

        # 5. Пересоздать RX стример и сразу запустить
        self._build_and_start_rx(self._rx_channel)

        # 6. Сдренировать мусор накопившийся пока RX был остановлен
        time.sleep(0.05)
        self._drain_rx(num_samples, count=8)

    def set_tx_freq(self, freq: float) -> None:
        self.usrp.set_tx_freq(
            self._uhd.types.TuneRequest(freq), self._tx_channel)

    def write_samples(self, iq_bytes: bytes, num_samples: int) -> None:
        """Отправить IQ в TX стример (непрерывный burst).

        Входной формат — int16 interleaved bytes (совместимость с BladeRF).
        start_of_burst выставляется только в первый вызов.
        """
        if self.tx_streamer is None:
            return

        # int16 interleaved → complex64 float
        raw = np.frombuffer(iq_bytes, dtype=np.int16).astype(np.float32)
        samples = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64) / 2048.0

        if self._tx_first_send:
            self._tx_md.start_of_burst = True
            self._tx_first_send = False
        else:
            self._tx_md.start_of_burst = False

        # send() сам блокируется до готовности буфера — sleep не нужен
        self.tx_streamer.send(samples, self._tx_md)

    def disable_tx(self) -> None:
        """Остановить TX и восстановить чистый RX стример.

        После совместной работы RX+TX в буфере RX стримера накапливается
        мусор и нарушается внутреннее состояние. Единственное надёжное
        решение — пересоздать RX стример заново.
        """
        # 1. Завершить TX burst
        self._teardown_tx_streamer()

        # 2. Пересоздать RX стример чтобы убрать мусор из буфера
        self._teardown_rx_streamer()
        self._build_and_start_rx(self._rx_channel)

        # 3. Сдренировать первые буферы с остатками от TX
        time.sleep(0.05)
        self._drain_rx(4096, count=8)

    def reconfigure_rx_after_tx(self, num_samples: int, config: SDRConfig) -> None:
        """Вызывается из UI после disable_tx когда идёт сканирование.

        disable_tx уже пересоздал стример. Здесь только дополнительный flush
        чтобы сегменты склеивались без пробелов.
        """
        self._drain_rx(num_samples, count=5)

    def get_actual_sample_rate(self) -> float:
        return float(self.usrp.get_rx_rate(self._rx_channel))

    def set_sample_rate(self, sample_rate: float, num_samples: int,
                        config: SDRConfig):
        """Изменить частоту дискретизации RX. TX стример не затрагивается."""
        current = self.usrp.get_rx_rate(self._rx_channel)
        if abs(current - sample_rate) < 1.0:
            return

        self._teardown_rx_streamer()
        self._sample_rate = sample_rate
        self.usrp.set_rx_rate(sample_rate, self._rx_channel)
        self.usrp.set_rx_bandwidth(sample_rate, self._rx_channel)
        self._build_and_start_rx(self._rx_channel)
        time.sleep(0.1)


# ============================================================================
# TX Thread
# ============================================================================

class TXThread(QThread):
    """Непрерывная TX передача в отдельном потоке."""

    def __init__(self, backend: SDRBackend, tx_buffer: bytes,
                 num_samples: int, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.tx_buffer = tx_buffer
        self.num_samples = num_samples
        self.running = True

    def run(self):
        while self.running:
            try:
                self.backend.write_samples(self.tx_buffer, self.num_samples)
                # Для USRP send() сам блокируется до готовности буфера.
                # Sleep здесь вызвал бы underflow.
            except Exception as e:
                print(f"TX send error: {e}")
                time.sleep(0.01)

    def stop(self):
        self.running = False
        self.wait()


# ============================================================================
# Main Spectrum Analyzer
# ============================================================================

class SpectrumAnalyzer(QMainWindow):
    """Main spectrum analyzer window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wideband Spectrum Analyzer (BladeRF 2.0 / USRP B205mini)")
        self.setGeometry(100, 100, 1400, 900)

        self.config = SDRConfig()
        self.backend: SDRBackend | None = None
        self.sdr_lock = Lock()

        self.current_rx_channel_index = 0
        self.current_tx_channel_index = 0

        self.live_scanning = False
        self.center_freq = 1000e6

        self.wb_centers = None
        self.wb_index = 0
        self.composite_spectrum = None
        self.common_freq = None

        self.maxhold_enabled = False
        self.maxhold_data_arr = None

        self.segment_correction = None
        self.trim_fraction = 0.0
        self.calibrating = False

        self.waterfall_data = None
        self.waterfall_index = 0

        self.tx_enabled = False
        self.tx_thread = None
        self.tx_buffer = None

        self.sweep_enabled = False
        self.sweep_timer = QTimer()
        self.sweep_timer.timeout.connect(self.next_sweep_step)
        self.sweep_freqs = []
        self.current_sweep_freq = 0

        self.current_x = None
        self.current_y = None

        self.iq_recording = False
        self.iq_save_path = None
        self.iq_accum = []

        self.init_ui()

    # =========================================================================
    # UI
    # =========================================================================

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ===== LEFT: Graphs =====
        graph_container = QWidget()
        graph_layout = QVBoxLayout(graph_container)

        self.graph_plot = pg.PlotWidget(title="Spectrum in dBm")
        self.graph_plot.setLabel('left', 'Power', units='dBm')
        self.graph_plot.setLabel('bottom', 'Frequency', units='MHz')
        self.graph_plot.getAxis('bottom').enableAutoSIPrefix(False)
        self.graph_plot.setYRange(-140, -30)
        self.graph_plot.showGrid(x=True, y=True)

        self.curve = self.graph_plot.plot(pen='y')
        self.maxhold_curve = self.graph_plot.plot(pen='r')

        self.max_marker = self.graph_plot.plot(
            symbol='o', symbolBrush='r', symbolSize=10)
        self.max_text = pg.TextItem(color='w', anchor=(0, 1))
        self.graph_plot.addItem(self.max_text)

        self.vLine = pg.InfiniteLine(angle=90, movable=False,
                                     pen=pg.mkPen('c'))
        self.hLine = pg.InfiniteLine(angle=0, movable=False,
                                     pen=pg.mkPen('c'))
        self.graph_plot.addItem(self.vLine, ignoreBounds=True)
        self.graph_plot.addItem(self.hLine, ignoreBounds=True)
        self.cursorLabel = pg.TextItem("", anchor=(1, 1),
                                       fill=pg.mkBrush(0, 0, 0, 150))
        self.graph_plot.addItem(self.cursorLabel)

        self.proxy = pg.SignalProxy(
            self.graph_plot.scene().sigMouseMoved,
            rateLimit=60, slot=self.on_mouse_moved)

        graph_layout.addWidget(self.graph_plot)

        self.waterfall_plot = pg.PlotWidget(title="Waterfall")
        self.waterfall_plot.setLabel('bottom', 'Frequency', units='MHz')
        self.waterfall_plot.setLabel('left', 'Scan')
        self.waterfall_plot.getAxis('bottom').enableAutoSIPrefix(False)
        self.waterfall_img = pg.ImageItem()
        self.waterfall_img.setOpts(invertY=False, axisOrder='row-major')
        lut = pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256)
        self.waterfall_img.setLookupTable(lut)
        self.waterfall_plot.addItem(self.waterfall_img)
        self.waterfall_plot.getViewBox().invertY(True)
        graph_layout.addWidget(self.waterfall_plot)

        # Waterfall color sliders
        wf_controls = QGroupBox("Waterfall Color Range")
        wf_layout = QHBoxLayout(wf_controls)

        wf_layout.addWidget(QLabel("Min:"))
        self.wf_min_label = QLabel("-140 dBm")
        self.wf_min_label.setMinimumWidth(70)
        self.wf_min_slider = QSlider(Qt.Horizontal)
        self.wf_min_slider.setRange(-180, 0)
        self.wf_min_slider.setValue(-140)
        self.wf_min_slider.valueChanged.connect(self.on_wf_range_changed)
        wf_layout.addWidget(self.wf_min_slider, stretch=1)
        wf_layout.addWidget(self.wf_min_label)
        wf_layout.addSpacing(20)

        wf_layout.addWidget(QLabel("Max:"))
        self.wf_max_label = QLabel("-30 dBm")
        self.wf_max_label.setMinimumWidth(70)
        self.wf_max_slider = QSlider(Qt.Horizontal)
        self.wf_max_slider.setRange(-180, 0)
        self.wf_max_slider.setValue(-30)
        self.wf_max_slider.valueChanged.connect(self.on_wf_range_changed)
        wf_layout.addWidget(self.wf_max_slider, stretch=1)
        wf_layout.addWidget(self.wf_max_label)

        wf_reset_btn = QPushButton("Reset")
        wf_reset_btn.setMaximumWidth(60)
        wf_reset_btn.clicked.connect(self.reset_wf_range)
        wf_layout.addWidget(wf_reset_btn)
        graph_layout.addWidget(wf_controls)

        main_layout.addWidget(graph_container, stretch=3)

        # ===== RIGHT: Control Panel =====
        control_panel = QWidget()
        control_layout = QFormLayout(control_panel)

        # --- Device Selection ---
        control_layout.addRow(QLabel("<b>Device</b>"))

        self.device_combo = QComboBox()
        self.device_combo.addItems(["BladeRF 2.0", "USRP B205mini"])
        self.device_combo.currentIndexChanged.connect(self.on_device_changed)
        control_layout.addRow("SDR Device:", self.device_combo)

        self.connect_button = QPushButton("Connect Device")
        self.connect_button.clicked.connect(self.connect_device)
        control_layout.addRow(self.connect_button)

        self.device_status_label = QLabel("Not connected")
        control_layout.addRow("Status:", self.device_status_label)

        sep0 = QFrame(); sep0.setFrameShape(QFrame.HLine)
        sep0.setFrameShadow(QFrame.Sunken)
        control_layout.addRow(sep0)

        # --- Receiver Settings ---
        control_layout.addRow(QLabel("<b>Receiver Settings</b>"))

        self.rx_channel_label = QLabel("RX Channel:")
        self.rx_channel_combo = QComboBox()
        self.rx_channel_combo.addItems(["RX1", "RX2"])
        self.rx_channel_combo.currentIndexChanged.connect(
            self.on_rx_channel_changed)
        control_layout.addRow(self.rx_channel_label, self.rx_channel_combo)

        self.start_freq_edit = QLineEdit("2300")
        control_layout.addRow("Start (MHz):", self.start_freq_edit)

        self.stop_freq_edit = QLineEdit("2500")
        control_layout.addRow("Stop (MHz):", self.stop_freq_edit)

        self.step_edit = QLineEdit("50")
        control_layout.addRow("Step (MHz):", self.step_edit)

        self.samples_combo = QComboBox()
        for v in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
            self.samples_combo.addItem(str(v))
        self.samples_combo.setCurrentText("4096")
        control_layout.addRow("FFT size:", self.samples_combo)

        self.gain_spin = QSpinBox()
        self.gain_spin.setRange(0, 76)
        self.gain_spin.setValue(30)
        control_layout.addRow("Gain:", self.gain_spin)

        self.waterfall_lines_spin = QSpinBox()
        self.waterfall_lines_spin.setRange(10, 2000)
        self.waterfall_lines_spin.setValue(30)
        control_layout.addRow("Waterfall size:", self.waterfall_lines_spin)

        self.scan_button = QPushButton("Start Scanning")
        self.scan_button.clicked.connect(self.toggle_live_scanning)
        control_layout.addRow(self.scan_button)

        self.maxhold_button = QPushButton("Turn Max Hold On")
        self.maxhold_button.clicked.connect(self.toggle_maxhold)
        control_layout.addRow(self.maxhold_button)

        self.calibrate_button = QPushButton("Calibrate Profile")
        self.calibrate_button.clicked.connect(self.run_calibration)
        control_layout.addRow(self.calibrate_button)

        self.calibrate_status_label = QLabel("Not calibrated")
        control_layout.addRow("Cal status:", self.calibrate_status_label)

        self.calibrate_clear_button = QPushButton("Clear Calibration")
        self.calibrate_clear_button.clicked.connect(self.clear_calibration)
        control_layout.addRow(self.calibrate_clear_button)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.HLine)
        sep1.setFrameShadow(QFrame.Sunken)
        control_layout.addRow(sep1)

        # --- TX Settings ---
        control_layout.addRow(QLabel("<b>Transmitter Settings</b>"))

        self.tx_channel_label = QLabel("TX Channel:")
        self.tx_channel_combo = QComboBox()
        self.tx_channel_combo.addItems(["TX1", "TX2"])
        self.tx_channel_combo.currentIndexChanged.connect(
            self.on_tx_channel_changed)
        control_layout.addRow(self.tx_channel_label, self.tx_channel_combo)

        self.tx_freq_edit = QLineEdit("2400")
        control_layout.addRow("TX Frequency (MHz):", self.tx_freq_edit)

        self.tx_gain_spin = QSpinBox()
        self.tx_gain_spin.setRange(0, 89)
        self.tx_gain_spin.setValue(10)
        control_layout.addRow("TX Gain:", self.tx_gain_spin)

        self.tx_start_button = QPushButton("Start Transmission")
        self.tx_start_button.clicked.connect(self.start_transmission)
        control_layout.addRow(self.tx_start_button)

        self.tx_status_label = QLabel("")
        control_layout.addRow("Status:", self.tx_status_label)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        sep2.setFrameShadow(QFrame.Sunken)
        control_layout.addRow(sep2)

        # --- Sweep Settings ---
        control_layout.addRow(QLabel("<b>Sweep Mode</b>"))

        self.sweep_start_edit = QLineEdit("2300")
        control_layout.addRow("Sweep Start (MHz):", self.sweep_start_edit)

        self.sweep_stop_edit = QLineEdit("2500")
        control_layout.addRow("Sweep Stop (MHz):", self.sweep_stop_edit)

        self.sweep_step_edit = QLineEdit("10")
        control_layout.addRow("Sweep Step (MHz):", self.sweep_step_edit)

        self.sweep_dwell_edit = QLineEdit("100")
        control_layout.addRow("Dwell Time (ms):", self.sweep_dwell_edit)

        self.sweep_gain_spin = QSpinBox()
        self.sweep_gain_spin.setRange(0, 89)
        self.sweep_gain_spin.setValue(30)
        control_layout.addRow("Sweep Gain:", self.sweep_gain_spin)

        self.sweep_start_button = QPushButton("Start Sweep")
        self.sweep_start_button.clicked.connect(self.toggle_sweep_transmission)
        control_layout.addRow(self.sweep_start_button)

        self.sweep_status_label = QLabel("")
        control_layout.addRow("Sweep Status:", self.sweep_status_label)

        sep3 = QFrame(); sep3.setFrameShape(QFrame.HLine)
        sep3.setFrameShadow(QFrame.Sunken)
        control_layout.addRow(sep3)

        # --- IQ Recording ---
        control_layout.addRow(QLabel("<b>IQ Data Recording</b>"))

        self.iq_freq_edit = QLineEdit("2400")
        control_layout.addRow("Center Freq (MHz):", self.iq_freq_edit)

        self.iq_duration_spin = QSpinBox()
        self.iq_duration_spin.setRange(1, 60000)
        self.iq_duration_spin.setValue(2)
        self.iq_duration_spin.setSuffix(" ms")
        control_layout.addRow("Duration:", self.iq_duration_spin)

        self.iq_record_button = QPushButton("Record && Save IQ...")
        self.iq_record_button.clicked.connect(self.start_iq_recording)
        control_layout.addRow(self.iq_record_button)

        self.iq_status_label = QLabel("Ready")
        control_layout.addRow("IQ Status:", self.iq_status_label)

        # Wrap control panel in scroll area
        scroll = QScrollArea()
        scroll.setWidget(control_panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        main_layout.addWidget(scroll, stretch=1)

    # =========================================================================
    # Device Management
    # =========================================================================

    def on_device_changed(self, index: int):
        self._update_channel_visibility()

    def _update_channel_visibility(self):
        device_name = self.device_combo.currentText()
        is_bladerf = "BladeRF" in device_name
        self.rx_channel_label.setVisible(is_bladerf)
        self.rx_channel_combo.setVisible(is_bladerf)
        self.tx_channel_label.setVisible(is_bladerf)
        self.tx_channel_combo.setVisible(is_bladerf)

        if is_bladerf:
            self.gain_spin.setRange(0, 60)
            self.tx_gain_spin.setRange(0, 60)
            self.sweep_gain_spin.setRange(0, 60)
        else:
            self.gain_spin.setRange(0, 76)
            self.tx_gain_spin.setRange(0, 89)
            self.sweep_gain_spin.setRange(0, 89)

    def connect_device(self):
        if self.backend is not None:
            self._disconnect_device()

        device_name = self.device_combo.currentText()
        self.device_status_label.setText("Connecting...")
        QApplication.processEvents()

        try:
            if "BladeRF" in device_name:
                self.backend = BladeRFBackend()
            else:
                self.backend = USRPBackend()

            self.backend.open()

            with self.sdr_lock:
                self.backend.configure_rx(
                    channel=self.current_rx_channel_index,
                    freq=self.center_freq,
                    sample_rate=self.config.sample_rate,
                    gain=self.config.gain,
                    num_samples=self.config.num_samples,
                    config=self.config,
                )

            self.device_status_label.setText(f"Connected: {device_name}")
            self.connect_button.setText("Disconnect")
            self.connect_button.clicked.disconnect()
            self.connect_button.clicked.connect(self._disconnect_device)
            print(f"Connected to {device_name}")

        except Exception as e:
            self.backend = None
            self.device_status_label.setText(f"Error: {e}")
            QMessageBox.critical(self, "Connection Error",
                                 f"Failed to connect to {device_name}:\n{e}")
            import traceback
            traceback.print_exc()

    def _disconnect_device(self):
        if self.live_scanning:
            self.live_scanning = False
            self.scan_button.setText("Start Scanning")
        if self.tx_thread is not None:
            self.tx_thread.stop()
            self.tx_thread = None
        self.sweep_timer.stop()
        self.tx_enabled = False
        self.sweep_enabled = False

        with self.sdr_lock:
            if self.backend is not None:
                self.backend.close()
                self.backend = None

        self.device_status_label.setText("Disconnected")
        self.connect_button.setText("Connect Device")
        self.connect_button.clicked.disconnect()
        self.connect_button.clicked.connect(self.connect_device)
        print("Device disconnected")

    def on_rx_channel_changed(self, index: int):
        if self.backend is None:
            return
        was_scanning = self.live_scanning
        if was_scanning:
            self.live_scanning = False
            self.scan_button.setText("Start Scanning")
        try:
            with self.sdr_lock:
                self.backend.configure_rx(
                    channel=index,
                    freq=self.center_freq,
                    sample_rate=self.config.sample_rate,
                    gain=self.config.gain,
                    num_samples=self.config.num_samples,
                    config=self.config,
                )
            self.current_rx_channel_index = index
            print(f"Switched to RX{index + 1}")
            if was_scanning:
                QTimer.singleShot(300, lambda: self.scan_button.click())
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to switch RX channel: {e}")

    def on_tx_channel_changed(self, index: int):
        if self.backend is None:
            return
        if self.tx_enabled:
            self.start_transmission()
        if self.sweep_enabled:
            self.toggle_sweep_transmission()
        self.current_tx_channel_index = index

    # =========================================================================
    # Spectrum Acquisition
    # =========================================================================

    def acquire_one_spectrum(self) -> tuple:
        with self.sdr_lock:
            try:
                samples = self.backend.read_samples(self.config.num_samples)

                if len(samples) < self.config.num_samples:
                    samples = np.pad(samples,
                                     (0, self.config.num_samples - len(samples)))

                window = np.blackman(len(samples))
                window_power = np.sum(window ** 2)

                spectrum = np.fft.fftshift(np.fft.fft(samples * window))
                power = np.abs(spectrum) ** 2 / window_power

                cal_offset = get_calibration(self.center_freq)
                power_dbm = 10 * np.log10(power / 1e-3 + 1e-12) + cal_offset

                if (self.segment_correction is not None and
                        len(self.segment_correction) == len(power_dbm)):
                    power_dbm = power_dbm - self.segment_correction

                freqs = np.fft.fftshift(
                    np.fft.fftfreq(self.config.num_samples,
                                   d=1.0 / self.config.sample_rate))
                freq_axis = (freqs + self.center_freq) / 1e6

                return freq_axis, power_dbm

            except Exception as e:
                print(f"Error acquiring spectrum: {e}")
                freq_axis = np.linspace(0, 1, self.config.num_samples)
                return freq_axis, np.full(self.config.num_samples, -140.0)

    # =========================================================================
    # Calibration
    # =========================================================================

    def run_calibration(self):
        if self.backend is None:
            QMessageBox.warning(self, "Error", "Device not connected")
            return
        if self.live_scanning:
            QMessageBox.warning(self, "Error",
                                "Stop scanning before calibration")
            return

        self.calibrating = True
        self.calibrate_button.setEnabled(False)
        self.calibrate_status_label.setText("Calibrating...")
        QApplication.processEvents()

        CAL_AVERAGES = 200

        try:
            flush_count = (self.config.SYNC_NUM_BUFFERS *
                           self.config.BUFFER_SIZE_MULTIPLIER + 4)
            with self.sdr_lock:
                self.backend.flush_rx_buffer(self.config.num_samples, flush_count)

            accum = np.zeros(self.config.num_samples, dtype=np.float64)
            for i in range(CAL_AVERAGES):
                _, power_dbm = self.acquire_one_spectrum()
                accum += 10.0 ** (power_dbm / 10.0)
                if i % 20 == 0:
                    self.calibrate_status_label.setText(
                        f"Calibrating... {i}/{CAL_AVERAGES}")
                    QApplication.processEvents()

            avg_profile = 10.0 * np.log10(accum / CAL_AVERAGES)
            n = len(avg_profile)
            ref_level = np.median(avg_profile[n // 3: 2 * n // 3])
            self.segment_correction = avg_profile - ref_level

            peak_correction = max(abs(self.segment_correction.min()),
                                  abs(self.segment_correction.max()))
            edge_left  = self.segment_correction[:n // 8].mean()
            edge_right = self.segment_correction[-n // 8:].mean()

            print(f"Calibration done. Ref={ref_level:.1f} dBm  "
                  f"peak±{peak_correction:.1f} dB")
            self.calibrate_status_label.setText(
                f"OK  peak±{peak_correction:.1f} dB  "
                f"L{edge_left:+.1f} R{edge_right:+.1f} dB")

        except Exception as e:
            print(f"Calibration error: {e}")
            self.segment_correction = None
            self.calibrate_status_label.setText(f"Error: {e}")
        finally:
            self.calibrating = False
            self.calibrate_button.setEnabled(True)

    def clear_calibration(self):
        self.segment_correction = None
        self.calibrate_status_label.setText("Not calibrated")

    # =========================================================================
    # Scanning
    # =========================================================================

    def toggle_live_scanning(self):
        if self.live_scanning:
            self.live_scanning = False
            self.scan_button.setText("Start Scanning")
        else:
            if self.backend is None:
                QMessageBox.warning(self, "Error", "Device not connected")
                return
            try:
                self.config.num_samples = int(self.samples_combo.currentText())
                self.config.waterfall_lines = self.waterfall_lines_spin.value()
                self.config.gain = self.gain_spin.value()

                with self.sdr_lock:
                    self.backend.set_rx_gain(self.config.gain)

                self.waterfall_data = np.full(
                    (self.config.waterfall_lines, self.config.num_samples),
                    -140.0)
                self.waterfall_index = 0
                self.waterfall_img.setImage(self.waterfall_data,
                                            autoLevels=False,
                                            levels=(-140, -30))

                self.live_scanning = True
                self.scan_button.setText("Pause")
                self.init_live_scan_parameters()
                self.composite_scan_cycle()

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to start scanning: {e}")
                import traceback
                traceback.print_exc()
                self.live_scanning = False

    def init_live_scan_parameters(self):
        try:
            start_mhz = float(self.start_freq_edit.text())
            stop_mhz  = float(self.stop_freq_edit.text())
            step_mhz  = float(self.step_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid frequency values")
            self.live_scanning = False
            self.scan_button.setText("Start Scanning")
            return

        BLADERF_MAX_SR = 61.44e6
        BLADERF_MIN_SR = 0.521e6
        USRP_MAX_SR    = 56e6
        USRP_MIN_SR    = 0.2e6
        TRIM = 0.15

        is_usrp = isinstance(self.backend, USRPBackend)
        max_sr = USRP_MAX_SR if is_usrp else BLADERF_MAX_SR
        min_sr = USRP_MIN_SR if is_usrp else BLADERF_MIN_SR

        desired_sr = step_mhz * 1e6 / (1.0 - 2.0 * TRIM) * 1.05
        desired_sr = float(np.clip(desired_sr, min_sr, max_sr))

        with self.sdr_lock:
            self.backend.set_sample_rate(desired_sr, self.config.num_samples,
                                         self.config)
            actual_sr = self.backend.get_actual_sample_rate()
            self.config.sample_rate = actual_sr

        if actual_sr > step_mhz * 1e6:
            self.trim_fraction = (0.5 * (1.0 - (step_mhz * 1e6) / actual_sr)
                                  * 0.95)
        else:
            self.trim_fraction = 0.0

        print(f"Step={step_mhz} MHz | Requested SR={desired_sr/1e6:.3f} MHz | "
              f"Actual SR={actual_sr/1e6:.3f} MHz | trim={self.trim_fraction:.4f}")

        start_hz = start_mhz * 1e6
        stop_hz  = stop_mhz  * 1e6
        step_hz  = step_mhz  * 1e6

        self.wb_centers = np.arange(start_hz + step_hz / 2, stop_hz, step_hz)
        self.wb_index = 0

        common_res = 0.1
        num_points  = int(np.round((stop_mhz - start_mhz) / common_res)) + 1
        self.common_freq = np.linspace(start_mhz, stop_mhz, num_points)
        self.composite_spectrum = np.full(self.common_freq.shape, -140.0)

        self.segment_accum   = None
        self.segment_count   = 0
        self.segment_avg_len = 5 if is_usrp else 20

        self.waterfall_img.setRect(
            QRectF(start_mhz, 0, stop_mhz - start_mhz,
                   self.config.waterfall_lines))
        self.curve.clear()
        self.waterfall_data.fill(-140)
        self.waterfall_index = 0

        if self.segment_correction is not None:
            if len(self.segment_correction) != self.config.num_samples:
                self.segment_correction = None
                self.calibrate_status_label.setText(
                    "Recalibrate needed (FFT size changed)")

    def composite_scan_cycle(self):
        if not self.live_scanning:
            return

        if self.wb_index >= len(self.wb_centers):
            self.update_display()
            self.wb_index = 0
            QTimer.singleShot(0, self.composite_scan_cycle)
            return

        new_center = self.wb_centers[self.wb_index]

        with self.sdr_lock:
            self.backend.set_rx_freq(new_center)
            self.center_freq = new_center
            if isinstance(self.backend, USRPBackend):
                time.sleep(0.002)
                flush_count = 1
            else:
                flush_count = (self.config.SYNC_NUM_BUFFERS *
                               self.config.BUFFER_SIZE_MULTIPLIER + 4)
            self.backend.flush_rx_buffer(self.config.num_samples, flush_count)

        self.segment_accum = None
        self.segment_count = 0
        QTimer.singleShot(0, self.do_composite_measurement)

    def do_composite_measurement(self):
        if not self.live_scanning:
            return

        try:
            meas_freq, meas_power = self.acquire_one_spectrum()

            if self.segment_accum is None:
                self.segment_accum = meas_power.copy()
            else:
                self.segment_accum += meas_power
            self.segment_count += 1

            if self.segment_count < self.segment_avg_len:
                QTimer.singleShot(0, self.do_composite_measurement)
                return

            avg_power = self.segment_accum / self.segment_count

            trim = int(len(meas_freq) * self.trim_fraction)
            if trim > 0:
                meas_freq_use = meas_freq[trim:-trim]
                avg_power_use = avg_power[trim:-trim]
            else:
                meas_freq_use = meas_freq
                avg_power_use = avg_power

            mask = ((self.common_freq >= meas_freq_use[0]) &
                    (self.common_freq <= meas_freq_use[-1]))

            if np.any(mask):
                interp_power = np.interp(
                    self.common_freq[mask], meas_freq_use, avg_power_use,
                    left=-140, right=-140)
                self.composite_spectrum[mask] = interp_power

                if self.maxhold_enabled:
                    if self.maxhold_data_arr is None:
                        self.maxhold_data_arr = self.composite_spectrum.copy()
                    else:
                        self.maxhold_data_arr[mask] = np.maximum(
                            self.maxhold_data_arr[mask], interp_power)

        except Exception as e:
            print(f"Measurement error: {e}")

        self.wb_index += 1
        QTimer.singleShot(0, self.composite_scan_cycle)

    def update_display(self):
        try:
            self.curve.setData(self.common_freq, self.composite_spectrum)
            self.current_x = self.common_freq.copy()
            self.current_y = self.composite_spectrum.copy()

            if self.maxhold_enabled and self.maxhold_data_arr is not None:
                self.maxhold_curve.setData(self.common_freq,
                                            self.maxhold_data_arr)
            else:
                self.maxhold_curve.clear()

            if self.composite_spectrum.size > 0:
                idx = np.argmax(self.composite_spectrum)
                f = self.common_freq[idx]
                p = self.composite_spectrum[idx]
                self.max_marker.setData([f], [p])
                self.max_text.setText(f"{f:.2f} MHz\n{p:.1f} dBm")
                self.max_text.setPos(f, p)

            row_data = np.interp(
                np.linspace(self.common_freq[0], self.common_freq[-1],
                            self.config.num_samples),
                self.common_freq, self.composite_spectrum)

            if self.waterfall_index < self.config.waterfall_lines:
                self.waterfall_data[self.waterfall_index] = row_data
                self.waterfall_index += 1
            else:
                self.waterfall_data = np.roll(self.waterfall_data, -1, axis=0)
                self.waterfall_data[-1] = row_data

            self.waterfall_img.setImage(
                self.waterfall_data, autoLevels=False,
                levels=(self.wf_min_slider.value(), self.wf_max_slider.value()))

        except Exception as e:
            print(f"Display error: {e}")

    def on_wf_range_changed(self):
        wf_min = self.wf_min_slider.value()
        wf_max = self.wf_max_slider.value()
        if wf_min >= wf_max:
            if self.sender() == self.wf_min_slider:
                wf_min = wf_max - 1
                self.wf_min_slider.blockSignals(True)
                self.wf_min_slider.setValue(wf_min)
                self.wf_min_slider.blockSignals(False)
            else:
                wf_max = wf_min + 1
                self.wf_max_slider.blockSignals(True)
                self.wf_max_slider.setValue(wf_max)
                self.wf_max_slider.blockSignals(False)
        self.wf_min_label.setText(f"{wf_min} dBm")
        self.wf_max_label.setText(f"{wf_max} dBm")
        if self.waterfall_data is not None:
            self.waterfall_img.setImage(self.waterfall_data, autoLevels=False,
                                        levels=(wf_min, wf_max))

    def reset_wf_range(self):
        self.wf_min_slider.setValue(-140)
        self.wf_max_slider.setValue(-30)

    def toggle_maxhold(self):
        self.maxhold_enabled = not self.maxhold_enabled
        if self.maxhold_enabled:
            self.maxhold_button.setText("Turn Max Hold Off")
            if self.common_freq is not None:
                self.maxhold_data_arr = self.composite_spectrum.copy()
        else:
            self.maxhold_button.setText("Turn Max Hold On")
            self.maxhold_curve.clear()

    # =========================================================================
    # Transmission
    # =========================================================================

    def _make_tx_buffer(self) -> bytes:
        """Сгенерировать CW тон в формате int16 interleaved (совместимо с BladeRF и USRP)."""
        t = np.arange(self.config.num_samples)
        tone_freq = 1000
        signal = np.exp(1j * 2 * np.pi * tone_freq * t /
                        self.config.sample_rate)
        iq = np.empty(2 * len(signal), dtype=np.int16)
        iq[0::2] = np.clip(np.real(signal) * 2047,
                           -2048, 2047).astype(np.int16)
        iq[1::2] = np.clip(np.imag(signal) * 2047,
                           -2048, 2047).astype(np.int16)
        return iq.tobytes()

    def start_transmission(self):
        if not self.tx_enabled:
            if self.backend is None:
                QMessageBox.warning(self, "Error", "Device not connected")
                return
            try:
                tx_freq = float(self.tx_freq_edit.text()) * 1e6
                tx_gain = self.tx_gain_spin.value()

                with self.sdr_lock:
                    self.backend.configure_tx(
                        channel=self.current_tx_channel_index,
                        freq=tx_freq,
                        sample_rate=self.config.sample_rate,
                        gain=tx_gain,
                        num_samples=self.config.num_samples,
                        config=self.config,
                    )

                self.tx_buffer = self._make_tx_buffer()
                self.tx_thread = TXThread(self.backend, self.tx_buffer,
                                          self.config.num_samples)
                self.tx_thread.start()

                self.tx_enabled = True
                self.tx_start_button.setText("Stop Transmission")
                self.tx_status_label.setText(
                    f"TX{self.current_tx_channel_index + 1}: "
                    f"{tx_freq/1e6:.2f} MHz, Gain {tx_gain}")
                print(f"TX started at {tx_freq/1e6:.2f} MHz")

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to start TX: {e}")
                import traceback
                traceback.print_exc()
                self.tx_enabled = False
        else:
            self.tx_enabled = False
            if self.tx_thread is not None:
                self.tx_thread.stop()
                self.tx_thread = None

            with self.sdr_lock:
                self.backend.disable_tx()
                if self.live_scanning:
                    self.backend.reconfigure_rx_after_tx(
                        self.config.num_samples, self.config)

            self.tx_start_button.setText("Start Transmission")
            self.tx_status_label.setText("Transmission stopped")
            print("TX stopped")

    def toggle_sweep_transmission(self):
        if not self.sweep_enabled:
            if self.backend is None:
                QMessageBox.warning(self, "Error", "Device not connected")
                return
            try:
                start_freq = float(self.sweep_start_edit.text()) * 1e6
                stop_freq  = float(self.sweep_stop_edit.text()) * 1e6
                step_freq  = float(self.sweep_step_edit.text()) * 1e6
                dwell_time = float(self.sweep_dwell_edit.text())

                if step_freq <= 0:
                    raise ValueError("Step must be positive")

                if start_freq < stop_freq:
                    self.sweep_freqs = np.arange(start_freq, stop_freq + step_freq,
                                                 step_freq)
                else:
                    self.sweep_freqs = np.arange(start_freq, stop_freq - step_freq,
                                                 -step_freq)

                if len(self.sweep_freqs) == 0:
                    raise ValueError("Invalid sweep parameters")

                tx_gain = self.sweep_gain_spin.value()
                with self.sdr_lock:
                    self.backend.configure_tx(
                        channel=self.current_tx_channel_index,
                        freq=self.sweep_freqs[0],
                        sample_rate=self.config.sample_rate,
                        gain=tx_gain,
                        num_samples=self.config.num_samples,
                        config=self.config,
                    )

                self.tx_buffer = self._make_tx_buffer()
                if self.tx_thread is None or not self.tx_thread.isRunning():
                    self.tx_thread = TXThread(self.backend, self.tx_buffer,
                                              self.config.num_samples)
                    self.tx_thread.start()

                self.sweep_enabled = True
                self.current_sweep_freq = 0
                self.sweep_timer.start(int(dwell_time))

                self.sweep_start_button.setText("Stop Sweep")
                self.sweep_status_label.setText(
                    f"TX{self.current_tx_channel_index + 1} Sweep: "
                    f"{self.sweep_freqs[0]/1e6:.2f} MHz")
                print(f"Sweep started: {start_freq/1e6:.1f}-{stop_freq/1e6:.1f} MHz")

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to start sweep: {e}")
                import traceback
                traceback.print_exc()
                self.sweep_enabled = False
        else:
            self.sweep_enabled = False
            self.sweep_timer.stop()
            self.sweep_start_button.setText("Start Sweep")
            self.sweep_status_label.setText("Sweep stopped")

            if not self.tx_enabled and self.tx_thread is not None:
                self.tx_thread.stop()
                self.tx_thread = None
                with self.sdr_lock:
                    self.backend.disable_tx()
                    if self.live_scanning:
                        self.backend.reconfigure_rx_after_tx(
                            self.config.num_samples, self.config)
            print("Sweep stopped")

    def next_sweep_step(self):
        if not self.sweep_enabled:
            return
        self.current_sweep_freq = (self.current_sweep_freq + 1) % len(
            self.sweep_freqs)
        freq = self.sweep_freqs[self.current_sweep_freq]
        with self.sdr_lock:
            self.backend.set_tx_freq(freq)
        self.sweep_status_label.setText(
            f"TX{self.current_tx_channel_index + 1} Sweep: {freq/1e6:.2f} MHz")

    # =========================================================================
    # Cursor
    # =========================================================================

    def on_mouse_moved(self, evt):
        pos = evt[0]
        if (self.graph_plot.sceneBoundingRect().contains(pos) and
                self.current_x is not None):
            mouse_point = self.graph_plot.getViewBox().mapSceneToView(pos)
            x = mouse_point.x()
            self.vLine.setPos(x)
            self.hLine.setPos(mouse_point.y())
            idx = (np.abs(self.current_x - x)).argmin()
            self.cursorLabel.setText(
                f"{self.current_x[idx]:.2f} MHz\n{self.current_y[idx]:.1f} dBm")
            self.cursorLabel.setPos(self.current_x[idx], self.current_y[idx])

    # =========================================================================
    # IQ Recording
    # =========================================================================

    def start_iq_recording(self):
        if self.backend is None:
            QMessageBox.warning(self, "Error", "Device not connected")
            return
        if self.iq_recording:
            QMessageBox.warning(self, "Error", "Recording already in progress")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save IQ Data", os.path.expanduser("~"),
            "CSV complex (*.csv);;Raw int16 IQ binary (*.bin);;NumPy binary (*.npy)")
        if not filepath:
            return

        if filepath.endswith(".npy"):
            fmt = "npy"
        elif filepath.endswith(".csv"):
            fmt = "csv"
        else:
            fmt = "bin"
            if not filepath.endswith(".bin"):
                filepath += ".bin"

        try:
            center_hz = float(self.iq_freq_edit.text()) * 1e6
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid center frequency")
            return

        duration_ms = self.iq_duration_spin.value()
        reads_needed = max(1, int(np.ceil(
            self.config.sample_rate * duration_ms / 1000 / self.config.num_samples)))

        self.iq_save_path = filepath
        self.iq_fmt = fmt
        self.iq_center_hz = center_hz
        self.iq_reads_needed = reads_needed
        self.iq_reads_done = 0
        self.iq_accum = []
        self.iq_recording = True

        self.iq_record_button.setEnabled(False)
        self.iq_status_label.setText("Recording...")

        was_scanning = self.live_scanning
        if was_scanning:
            self.live_scanning = False

        with self.sdr_lock:
            self.backend.set_rx_freq(center_hz)
            self.center_freq = center_hz
            flush_count = (self.config.SYNC_NUM_BUFFERS *
                           self.config.BUFFER_SIZE_MULTIPLIER + 4)
            self.backend.flush_rx_buffer(self.config.num_samples, flush_count)

        self._iq_was_scanning = was_scanning
        QTimer.singleShot(0, self._iq_read_chunk)

    def _iq_read_chunk(self):
        if not self.iq_recording:
            return
        try:
            with self.sdr_lock:
                samples = self.backend.read_samples(self.config.num_samples)

            if len(samples) < self.config.num_samples:
                samples = np.pad(samples,
                                 (0, self.config.num_samples - len(samples)))

            self.iq_accum.append(samples.copy())
            self.iq_reads_done += 1
            progress = int(self.iq_reads_done / self.iq_reads_needed * 100)
            self.iq_status_label.setText(f"Recording... {progress}%")

            if self.iq_reads_done < self.iq_reads_needed:
                QTimer.singleShot(0, self._iq_read_chunk)
            else:
                self._iq_save()
        except Exception as e:
            print(f"IQ read error: {e}")
            self.iq_recording = False
            self.iq_record_button.setEnabled(True)
            self.iq_status_label.setText(f"Error: {e}")

    def _iq_save(self):
        try:
            all_samples = np.concatenate(self.iq_accum)
            filepath = self.iq_save_path
            fmt = self.iq_fmt

            if fmt == "npy":
                np.save(filepath, all_samples)
            elif fmt == "csv":
                data = np.column_stack([all_samples.real, all_samples.imag])
                np.savetxt(filepath, data, delimiter=",",
                           header="real,imag", comments="")
            else:
                iq_int16 = np.empty(len(all_samples) * 2, dtype=np.int16)
                iq_int16[0::2] = np.clip(all_samples.real * 2047,
                                          -2048, 2047).astype(np.int16)
                iq_int16[1::2] = np.clip(all_samples.imag * 2047,
                                          -2048, 2047).astype(np.int16)
                iq_int16.tofile(filepath)

            meta_path = filepath.rsplit(".", 1)[0] + "_meta.txt"
            with open(meta_path, "w") as f:
                f.write(f"center_freq_hz={self.iq_center_hz}\n")
                f.write(f"sample_rate_hz={self.config.sample_rate}\n")
                f.write(f"num_samples={len(all_samples)}\n")
                f.write(f"format={fmt}\n")
                f.write(f"gain={self.config.gain}\n")
                f.write(f"device={self.device_combo.currentText()}\n")
                f.write(f"recorded_at={datetime.now().isoformat()}\n")

            size_kb = os.path.getsize(filepath) / 1024
            print(f"IQ saved: {filepath} ({len(all_samples)} samples, "
                  f"{size_kb:.1f} KB)")
            self.iq_status_label.setText(
                f"Saved {len(all_samples)} samples ({size_kb:.0f} KB)")

        except Exception as e:
            print(f"IQ save error: {e}")
            self.iq_status_label.setText(f"Save error: {e}")
        finally:
            self.iq_recording = False
            self.iq_accum = []
            self.iq_record_button.setEnabled(True)
            if self._iq_was_scanning:
                QTimer.singleShot(200, lambda: self.scan_button.click())

    # =========================================================================
    # Close
    # =========================================================================

    def closeEvent(self, event):
        print("Closing application...")
        self.live_scanning = False
        if self.tx_thread is not None:
            self.tx_thread.stop()
        self.sweep_timer.stop()
        with self.sdr_lock:
            if self.backend is not None:
                self.backend.close()
        print("Device closed")
        event.accept()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 65)
    print("Wideband Spectrum Analyzer (BladeRF 2.0 / USRP B205mini)")
    print("=" * 65)

    QLocale.setDefault(QLocale("C"))

    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    log_filename = datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.txt")
    log_path = os.path.join(base_path, log_filename)
    logger = Logger(log_path)
    sys.stdout = logger
    sys.stderr = logger

    print(f"Log: {log_path}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")

    app = QApplication(sys.argv)

    if getattr(sys, 'frozen', False):
        icon_path = os.path.join(sys._MEIPASS, "bladerf2_0.ico")
    else:
        icon_path = os.path.join(base_path, "bladerf2_0.ico")

    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    try:
        window = SpectrumAnalyzer()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        try:
            QMessageBox.critical(None, "Fatal Error",
                                 f"Application failed to start:\n{e}")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()