# Corrupted Transmission Recovery

## Summary
I purified the `corrupted.wav` file using the functions of fft and filtering of python and some basic concepts. 

## Tools Used
- Python 3
- NumPy, SciPy, Matplotlib

---

## Stage 1 — Initial Inspection
The file is sampled at **44100 Hz**, duration **2.28 seconds**.

Time domain shows no visible speech structure.  
FFT shows all energy centered around **7300 Hz** — the normal 
speech band (0–4 kHz) is empty.

**Conclusion:** Signal has been shifted up in frequency.

---

## Stage 2 — Demodulation
The speech was AM modulated with a carrier at **7300.2 Hz** —  
multiplying the audio by `cos(2π × 7300 × t)` shifts all energy 
up to the carrier frequency.

**Fix:**
1. Multiply by same carrier → shifts energy back to 0–4 kHz
2. Low pass filter (Butterworth, cutoff 4500 Hz, order 6) → 
   removes the unwanted copy at 14600 Hz

---

## Stage 3 — Narrowband Interference Removal
After demodulation, FFT showed 3 razor-thin spikes — not typical 
of natural speech (which spreads energy broadly).

Each bin was compared against its local neighbourhood average.  
Bins 5x louder than neighbours were flagged as spikes.

**Spikes found:** 1200.1 Hz, 2199.9 Hz, 4100.1 Hz

**Fix:** IIR notch filter (Q=40) at each frequency — surgically 
removes only that frequency, leaves surrounding audio untouched.

---

## Stage 4 — DC Offset Removal
FFT showed a spike at exactly **0 Hz** — signature of a DC offset.  
Signal mean confirmed: **DC offset = 6028.7** (on ±32767 scale).

A DC offset shifts the waveform baseline away from zero causing 
asymmetric clipping and distortion.

**Fix:** Subtracted the mean from every sample.
```python
stage4_signal = stage3_signal - stage3_signal.mean()
```

---

## What Was Done to the Signal

| Stage | Corruption | Fix |
|-------|-----------|-----|
| 2 | AM modulation at 7300 Hz | Multiply by carrier + low pass filter |
| 3 | Sine tones at 1200, 2200, 4100 Hz | Notch filters (Q=40) |
| 4 | DC offset of +6028 | Subtract mean |

---

## Files
- `corrupted.wav` — original file (unchanged)
- `recovered.wav` — final clean output
- `solution.py` — complete solution
- `plots/` — all intermediate and final plots