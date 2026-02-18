# EE217 Homework 3 - Solutions Summary

## Problem 1: PRBS Signaling

### Part a) PRBS Generation
Successfully generated all four PRBS sequences:
- **PRBS7**: 127 bits (2^7 - 1)
- **PRBS127**: 127 bits (same as PRBS7, uses polynomial 0x60)
- **PRBS511**: 511 bits (2^9 - 1, polynomial 0x110)
- **PRBS1023**: 1023 bits (2^10 - 1, polynomial 0x240)

Code implementation uses a Galois LFSR (Linear Feedback Shift Register).

### Part b) Autocorrelation Analysis

**PRBS511 Autocorrelation:**
- Peak at zero offset: **511** ✓
- Off-peak values: **-1** ✓
- This confirms the PRBS511 sequence was generated correctly

**255-bit Subsequence Analysis:**
- **Answer: NO**, a 255-bit subsequence of PRBS511 is NOT a true PRBS255
- **Best subsequence starts at index**: 292
- **Maximum off-peak autocorrelation**: 27 (Expected: -1 for true PRBS255)
- **It DOES depend on which subsequence you choose** - different starting positions give different autocorrelation properties

**Why it's not a true PRBS255:**
A true PRBS255 would require a maximal-length 8-bit LFSR sequence. A subsequence of PRBS511 doesn't have the proper feedback polynomial structure to be a maximal-length 255-bit sequence, so its autocorrelation properties differ from a true PRBS255.

### Part c) Cross-correlation (Extra Credit)

**Comparison of two PRBS511 generators:**
- Polynomial A: 0x110 (x^9 + x^5 + 1)
- Polynomial B: 0x108 (x^9 + x^4 + 1)

**Results:**
- **seqA equals seqB:** NO
- **Maximum cross-correlation:** 45
- The sequences are different but share some correlation structure
- Both generate valid PRBS511 sequences with period 511

**See plot:** `problem1c_crosscorr.png`

---

## Problem 2: CDMA-Based Touch Sensors

### Part a) Correlation Peak Analysis

**Setup:**
- 5 drive lines with PRBS511 modulation at different phase offsets
- Initial LFSR state: 257 (binary: 0b100000001)
- Polynomial: 0x110

**Drive/Sense Pair Measurements (No Touch):**

| Drive Line | Phase Offset (samples) | Relative Capacitance |
|------------|------------------------|----------------------|
| 1 (5mm)    | 21                     | 39.3432              |
| 2 (10mm)   | 151                    | 27.1221              |
| 3 (15mm)   | 276                    | 33.7101              |
| 4 (20mm)   | 341                    | 15.7223              |
| 5 (25mm)   | 432                    | 7.1733               |

### Part b) Touch Location Detection

**Capacitance Changes Due to Touch:**

| Position | No Touch | With Touch | Difference  |
|----------|----------|------------|-------------|
| 5 mm     | 39.3432  | 34.9040    | -4.4391     |
| 10 mm    | 27.1221  | 10.4966    | **-16.6255**|
| 15 mm    | 33.7101  | 25.0939    | -8.6163     |
| 20 mm    | 15.7223  | 13.9834    | -1.7389     |
| 25 mm    | 7.1733   | 8.1044     | +0.9311     |

**Touch Location Estimation:**
- **Estimated touch location:** 10.86 mm
- **Touch amplitude:** -16.97 (indicating capacitance decrease)
- **Touch spread (sigma):** 3.63 mm (Gaussian width)

The touch was detected near the 10mm position, with maximum capacitance change at that location. The Gaussian fit shows the touch affects primarily the second sensor with decreasing effect on neighboring sensors.

**See plot:** `problem2_touch_sensor.png`

### Part c) Noise Estimation (Extra Credit)

**Noise Analysis:**
- **Standard deviation of off-peak correlation:** 2.19
- **Estimated noise per sample (before correlation):** 0.097

The correlation process provides processing gain equal to the square root of the PRBS length (√511 ≈ 22.6), which reduces the effective noise by this factor.

---

## Problem 3: DFT/FFT Exercises

### Part a) Relationship between Y(k) and H(ω)

**Answer:**
For a linear time-invariant (LTI) system with frequency response H(ω) excited by periodic input x(n) with period N:

**Y(k) = H(2πk/N) · X(k)**

Where:
- Y(k) is the N-point DFT of the output y(n)
- X(k) is the N-point DFT of the input x(n)
- H(ω) is the frequency response of the system

**Explanation:**
The periodic input can be decomposed into frequency components at discrete frequencies ω = 2πk/N. Each frequency component is scaled by the system's frequency response H(ω) evaluated at that frequency. The DFT Y(k) represents samples of the output spectrum at these discrete points.

### Part b) Minimum Sampling Requirements

**Given:**
- Bandwidth B = 4 kHz
- Desired frequency resolution Δf ≤ 50 Hz
- N = 2^m (power of 2)

**Answers:**

1. **Minimum sampling rate (Nyquist):**
   - fs_min = 2B = **8,000 Hz** or **8 kHz**

2. **Minimum record length:**
   - T_min = 1/Δf = 1/50 = **0.02 seconds** or **20 ms**

3. **Minimum number of samples:**
   - N_min = fs_min × T_min = 8000 × 0.02 = 160 samples
   - Since N = 2^m, round up to: **N = 256 samples (2^8)**

### Part c) Frequency Sampling and Time-Domain Aliasing

**Signal:** x(n) = 0.8^|n|

**DTFT:** X(ω) = (1 - 0.64) / (1 - 1.6cos(ω) + 0.64)

**Results for N=20:**
The reconstructed signal shows significant deviation from the true signal due to **time-domain aliasing** (circular time-aliasing).

**Results for N=100:**
Much better reconstruction near n=0, with minimal aliasing effects.

**What's happening when N=20?**

The true signal x(n) = 0.8^|n| extends infinitely in both directions. When we:
1. Sample the DTFT at N discrete frequencies
2. Take the IFFT

We get a **PERIODIC time-domain signal with period N**.

For N=20:
- The signal tail at n > 10 wraps around and appears at n < 0
- The signal tail at n < -10 wraps around and appears at n > 0
- This aliasing causes significant distortion in the reconstructed signal

For N=100:
- The period is much longer (100 samples)
- The signal has decayed to nearly zero at n = ±50
- Less aliasing occurs, giving accurate reconstruction near n = 0

**Key Insight:**
This is analogous to frequency-domain aliasing in sampling, but occurring in the **time domain due to frequency-domain sampling**! Just as time-domain sampling can cause frequency aliasing if the sampling rate is too low, frequency-domain sampling can cause time-domain aliasing if we don't sample enough frequency points.

**See plots:**
- `problem3c_dtft.png` - The DTFT of x(n)
- `problem3c_ifft_comparison.png` - Comparison of N=20 vs N=100 reconstructions

---

## Summary of Generated Files

All code is in: `hw3_solution.py`

All plots:
1. `problem1b_autocorr_prbs511.png` - Autocorrelation of PRBS511
2. `problem1b_autocorr_255bit.png` - Best 255-bit subsequence autocorrelation
3. `problem1c_crosscorr.png` - Cross-correlation between two PRBS511 sequences
4. `problem2_touch_sensor.png` - Touch sensor analysis (correlations and location)
5. `problem3c_dtft.png` - DTFT plot
6. `problem3c_ifft_comparison.png` - Time-domain aliasing demonstration

---

## Key Concepts Demonstrated

1. **Linear Feedback Shift Registers (LFSRs):** Galois implementation for PRBS generation
2. **Autocorrelation:** Properties of maximal-length sequences
3. **CDMA (Code Division Multiple Access):** Using orthogonal codes for simultaneous signaling
4. **Signal Processing:** Correlation-based detection and noise analysis
5. **DFT/FFT Theory:** Sampling in time and frequency domains, aliasing effects
6. **Touch Sensing:** Capacitive sensing with PRBS modulation
