#step 1
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import os

os.makedirs('plots',exist_ok=True)

sr, data=wavfile.read(r'C:\Users\91975\OneDrive\Documents\GitHub\ERC_assignment\Q3\corrupted.wav')

if data.ndim>1:
    data = data[:,0]

signal=data.astype(np.float32)
n=len(signal)
t=np.arange(n)/sr

print(f"Sample rate  : {sr} Hz")
print(f"No. of samples : {n}")
print(f"Duration     : {n/sr:.2f} seconds")

#time domain conversion of signal(graph)
plt.figure(figsize=(12, 4))
plt.plot(t, signal, color='steelblue', linewidth=0.5)
plt.title('Stage 1 — Time Domain of Corrupted Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/stage1_time_domain.png', dpi=150)
plt.show()


freqs= fftfreq (n ,1/sr)
fft_mag = np.abs(fft(signal))

index= freqs>0
freqs_pos =freqs[index]
mag_pos =fft_mag[index]

peak_freq = freqs_pos[np.argmax(mag_pos)]
print(f"\nPeak energy at : {peak_freq:.1f} Hz")

#fourier tranform of the signal
plt.figure(figsize=(12, 4))
plt.plot(freqs_pos, mag_pos, color='darkorange', linewidth=0.5)
plt.axvline(peak_freq, color='red', linestyle='--', linewidth=1.2,
            label=f'Peak at {peak_freq:.0f} Hz')
plt.title('Stage 1 — FFT of Corrupted Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, sr//2)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/stage1_fft.png', dpi=150)
plt.show()



# step 2
from scipy.signal import butter , filtfilt

peak_f=peak_freq
modulator=np.cos(2*(np.pi)*t*peak_f)
demodulated= signal* modulator

#signal in time domain at stage 2
plt.figure(figsize=(12, 4))
plt.plot(t, demodulated, color='purple', linewidth=0.5)
plt.title('Stage 2 — Time Domain After Multiplying by modulator')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/stage2_time_domain_intermediate.png', dpi=150)
plt.show()

#function to remove high frequencies(low pass filter)
def lowpass_filter(sig, cutoff, fs, order=6):
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, sig)

#applying low pass filter
stage2_signal=lowpass_filter(demodulated, 4500, sr)

fft_stage2_signal = np.abs(fft(stage2_signal))
freqs_stage2_signal= fftfreq(n,1/sr)

index2= freqs_stage2_signal >0
freqs2_pos = freqs_stage2_signal[index2]
mag2_pos =fft_stage2_signal[index2]


#graph after clearing the higher frequencies that where present in the signal
plt.figure(figsize=(12, 4))
plt.plot(freqs2_pos, mag2_pos, color='steelblue', linewidth=0.5)
plt.axvspan(0, 4000, alpha=0.15, color='green', label='Speech band (0–4 kHz)')
plt.title('Stage 2 — FFT After Demodulation')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 8000)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/stage2_fft_after_demod.png', dpi=150)
plt.show()



#step 3 

#finding dominanting frequencies that are of no need 

spike_freqs=[]
for i in range(len(freqs2_pos)):
    if freqs2_pos[i] >0  and freqs2_pos[i]<4500:
        if  mag2_pos[i] > 5* np.mean(mag2_pos[max(0,i-5) : i+5]):
            print(f"Spike at {freqs2_pos[i]:.1f}")
            spike_freqs.append(freqs2_pos[i])



#removing them
from scipy.signal import iirnotch

stage3_signal=stage2_signal.copy()

for freq in spike_freqs:
    b, a = iirnotch(freq, Q=40, fs=sr)
    stage3_signal = filtfilt(b, a, stage3_signal)

fft_stage3_signal = np.abs(fft(stage3_signal))
freqs_stage3_signal= fftfreq(n,1/sr)

index3= freqs_stage3_signal >0
freqs3_pos = freqs_stage3_signal[index3]
mag3_pos =fft_stage3_signal[index3]

#graph after removing the noises
plt.figure(figsize=(12, 4))
plt.plot(freqs3_pos, mag3_pos, color='steelblue', linewidth=0.5)
plt.axvspan(0, 4000, alpha=0.15, color='green', label='Speech band (0–4 kHz)')
plt.title('Stage 3 — FFT After Notch Filtering')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 5000)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/stage3_fft_after_notch.png', dpi=150)
plt.show()


#step 4
#checking dc shift (got a hint from spike near 0 hz)
print (f" DC offset :{stage3_signal.mean():.4f}")

#found dc offset in the signal as there is a spike at 0hz
plt.figure(figsize=(12, 4))
plt.plot(t, stage3_signal, color='red', linewidth=0.5)
plt.axhline(stage3_signal.mean(), color='black', linestyle='--',
            linewidth=1.5, label=f'DC offset = {stage3_signal.mean():.1f}')
plt.axhline(0, color='green', linestyle='--',
            linewidth=1, label='Zero line')
plt.title('Stage 4 — DC Offset Problem ')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/stage4_time domain_dc_offset.png', dpi=150)
plt.show()

stage4_signal=stage3_signal-stage3_signal.mean()

#after removal of dc offset
plt.figure(figsize=(12, 4))
plt.plot(t, stage4_signal, color='red', linewidth=0.5)
plt.axhline(stage4_signal.mean(), color='black', linestyle='--',
            linewidth=1.5, label=f'DC offset = {stage4_signal.mean():.1f}')
plt.axhline(0, color='green', linestyle='--',
            linewidth=1, label='Zero line')
plt.title('Stage 4 — DC Offset solved ')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/stage4_time domain_after_completing.png', dpi=150)
plt.show()


fft_stage4_signal = np.abs(fft(stage4_signal))
freqs_stage4_signal= fftfreq(n,1/sr)

index4= freqs_stage4_signal >0
freqs4_pos = freqs_stage4_signal[index4]
mag4_pos =fft_stage4_signal[index4]

#graph after complete working of fft
plt.figure(figsize=(12, 4))
plt.plot(freqs4_pos, mag4_pos, color='steelblue', linewidth=0.5)
plt.axvspan(0, 4000, alpha=0.15, color='green', label='Speech band (0–4 kHz)')
plt.title('Stage 4 — FFT After removing DC offset')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 5000)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/stage4_fft_final_purifying.png', dpi=150)
plt.show()


recovered_signal= stage4_signal / np. max(np.abs(stage4_signal))
wavfile.write('recovered.wav',sr,(recovered_signal*32767).astype(np.int16))

