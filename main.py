from bladerf import _bladerf
import numpy as np
import matplotlib.pyplot as plt
import time

# BladeRF инициализация
sdr = _bladerf.BladeRF()
rx = sdr.Channel(_bladerf.CHANNEL_RX(0))  # RX1

rx.frequency = int(5100e6)        # Проверь, генератор точно здесь?
rx.sample_rate = int(40e6)
rx.bandwidth = int(28e6)
rx.gain_mode = _bladerf.GainMode.Manual
rx.gain = 50
rx.enable = True

sdr.sync_config(
    layout=_bladerf.ChannelLayout.RX_X1,
    fmt=_bladerf.Format.SC16_Q11,
    num_buffers=16,
    buffer_size=32768 * 4,
    num_transfers=8,
    stream_timeout=3500
)

# Подождать немного
time.sleep(0.05)

# Захват
buf = bytearray(32768 * 4)
sdr.sync_rx(buf, 32768)
samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
samples = samples.view(np.complex64) / (2**11)

# Спектр
windowed = samples * np.hanning(len(samples))
spectrum = np.fft.fftshift(np.fft.fft(windowed))
power_db = 20 * np.log10(np.abs(spectrum) / len(samples) + 1e-12)
freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), d=1/40e6)) + 1000e6

plt.plot(freqs / 1e6, power_db)
plt.title("Тест сигнала на 1000 МГц")
plt.xlabel("Частота (МГц)")
plt.ylabel("Амплитуда (dBFS)")
plt.grid(True)
plt.tight_layout()
plt.show()

rx.enable = False
sdr.close()
