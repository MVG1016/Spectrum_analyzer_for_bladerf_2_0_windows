from bladerf import _bladerf
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000  # или даже 5000, если будет снова падать


# --- Параметры ---
START_FREQ = 100e6
END_FREQ = 6000e6
SPAN = 25e6           # ширина захвата
STEP = 20e6           # шаг с перекрытием
SAMPLE_RATE = 40e6
NUM_SAMPLES = 4096
GAIN = 20
AVG_COUNT = 4         # Количество усреднений на одной частоте

# --- SDR инициализация ---
sdr = _bladerf.BladeRF()
rx = sdr.Channel(_bladerf.CHANNEL_RX(0))

rx.sample_rate = int(SAMPLE_RATE)
rx.bandwidth = int(SPAN)
rx.gain_mode = _bladerf.GainMode.Manual
rx.gain = GAIN
rx.enable = True

sdr.sync_config(
    layout=_bladerf.ChannelLayout.RX_X1,
    fmt=_bladerf.Format.SC16_Q11,
    num_buffers=16,
    buffer_size=NUM_SAMPLES * 4,
    num_transfers=8,
    stream_timeout=3500
)

# --- Частотные точки ---
center_freqs = np.arange(START_FREQ, END_FREQ, STEP)
full_spectrum = []
full_freqs = []

print(f"\nНачинаю сканирование от {START_FREQ/1e6:.0f} до {END_FREQ/1e6:.0f} МГц...")

start_time = time.time()

for freq in center_freqs:
    #print(f"→ Частота {freq/1e6:.1f} МГц")
    rx.frequency = int(freq)
    time.sleep(0.01)

    avg_power = None

    for _ in range(AVG_COUNT):
        buf = bytearray(NUM_SAMPLES * 4)
        sdr.sync_rx(buf, NUM_SAMPLES)

        samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
        samples = samples.view(np.complex64) / (2**11)

        # Окно и FFT
        windowed = samples * np.hanning(len(samples))
        spectrum = np.fft.fftshift(np.fft.fft(windowed))
        power_db = 20 * np.log10(np.abs(spectrum) / len(samples) + 1e-12)

        if avg_power is None:
            avg_power = power_db
        else:
            avg_power += power_db

    avg_power /= AVG_COUNT  # Усреднение

    freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), d=1/SAMPLE_RATE)) + freq

    full_spectrum.append(avg_power)
    full_freqs.append(freqs)

end_time = time.time()
print(f"\nСканирование завершено за {end_time - start_time:.2f} секунд.")

# --- Объединение ---
full_spectrum = np.concatenate(full_spectrum)
full_freqs = np.concatenate(full_freqs)

# --- График ---
plt.figure(figsize=(16, 6))
plt.plot(full_freqs / 1e6, full_spectrum, linewidth=0.5)
plt.title(f"BladeRF 2.0 — Спектр 100–6000 МГц (усреднение: {AVG_COUNT}x)")
plt.xlabel("Частота (МГц)")
plt.ylabel("Амплитуда (dBFS)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Завершение ---
rx.enable = False
sdr.close()