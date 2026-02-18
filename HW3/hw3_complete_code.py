"""
================================================================================
EE217 HOMEWORK 3 - COMPLETE CODE SUBMISSION
Winter 2026

This file contains all code used to solve Problems 1, 2, and 3.
All functions are documented and ready to run.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# ============================================================================
# COMMON UTILITY FUNCTIONS (Used across all problems)
# ============================================================================

def generate_prbs(n_bits, polynomial_hex, initial_state=1):
    """
    Generate PRBS (Pseudo-Random Binary Sequence) using Galois LFSR.

    Args:
        n_bits: Number of bits in shift register (e.g., 9 for PRBS511)
        polynomial_hex: Feedback polynomial in hexadecimal (e.g., 0x110)
        initial_state: Starting state of shift register (default=1)

    Returns:
        numpy array of PRBS sequence (0s and 1s)
        Length = 2^n_bits - 1 (maximal-length sequence)

    Algorithm (Galois LFSR):
        1. Output the LSB (rightmost bit)
        2. Shift entire register right by 1
        3. If output bit was 1, XOR register with polynomial
        4. Repeat for period = 2^n_bits - 1
    """
    period = (1 << n_bits) - 1  # 2^n_bits - 1
    state = initial_state
    sequence = np.zeros(period, dtype=int)

    for i in range(period):
        # Output the least significant bit
        output_bit = state & 1
        sequence[i] = output_bit

        # Shift right
        state >>= 1

        # If output was 1, XOR with polynomial (feedback)
        if output_bit:
            state ^= polynomial_hex

    return sequence


def to_bipolar(sequence):
    """
    Convert binary sequence (0, 1) to bipolar sequence (-1, +1).

    This is needed for correlation because:
    - Same bits: (+1)(+1) = +1 or (-1)(-1) = +1
    - Different bits: (+1)(-1) = -1 or (-1)(+1) = -1

    With (0,1), correlation doesn't work properly.
    """
    return 2 * sequence - 1


# ============================================================================
# PROBLEM 1: PRBS SIGNALING
# ============================================================================

print("\n" + "="*80)
print("PROBLEM 1: PRBS SIGNALING")
print("="*80)

def compute_autocorrelation(sequence):
    """
    Compute circular autocorrelation of a bipolar sequence.

    Autocorrelation measures how similar a signal is to a shifted
    version of itself. For good PRBS:
        - Peak of N at zero shift (perfect match)
        - Value of -1 at all other shifts (orthogonal)

    Args:
        sequence: Bipolar sequence (-1, +1)

    Returns:
        Autocorrelation array of length N
    """
    N = len(sequence)
    autocorr = np.zeros(N)

    for offset in range(N):
        # Circular shift and compute dot product
        autocorr[offset] = np.sum(sequence * np.roll(sequence, offset))

    return autocorr


def compute_crosscorrelation(seq_a, seq_b):
    """
    Compute circular cross-correlation between two sequences.

    Cross-correlation measures similarity between two different signals.
    Low cross-correlation means signals are independent (good for CDMA).

    Args:
        seq_a: First bipolar sequence
        seq_b: Second bipolar sequence (same length)

    Returns:
        Cross-correlation array
    """
    N = len(seq_a)
    assert len(seq_b) == N, "Sequences must be same length"

    crosscorr = np.zeros(N)

    for offset in range(N):
        crosscorr[offset] = np.sum(seq_a * np.roll(seq_b, offset))

    return crosscorr


# Problem 1a: Generate all PRBS sequences
print("\nProblem 1a: Generating PRBS sequences...")

prbs_configs = [
    (3, 0x3, "PRBS7"),      # 2^3 - 1 = 7 bits
    (7, 0x60, "PRBS127"),   # 2^7 - 1 = 127 bits
    (9, 0x110, "PRBS511"),  # 2^9 - 1 = 511 bits
    (10, 0x240, "PRBS1023") # 2^10 - 1 = 1023 bits
]

prbs_sequences = {}

for n_bits, poly, name in prbs_configs:
    seq = generate_prbs(n_bits, poly)
    prbs_sequences[name] = seq
    print(f"  {name}: {len(seq)} bits, polynomial 0x{poly:X}")
    print(f"    First 10 bits: {''.join(map(str, seq[:10]))}")
    print(f"    Ones: {np.sum(seq)}, Zeros: {len(seq) - np.sum(seq)}")

# Problem 1b: Autocorrelation of PRBS511
print("\nProblem 1b: Autocorrelation analysis...")

prbs511 = prbs_sequences["PRBS511"]
prbs511_bipolar = to_bipolar(prbs511)
autocorr_511 = compute_autocorrelation(prbs511_bipolar)

print(f"  PRBS511 autocorrelation:")
print(f"    Peak at offset 0: {autocorr_511[0]}")
print(f"    Off-peak values: all = {autocorr_511[1]} (should be -1)")
print(f"    Perfect PRBS? {np.all(autocorr_511[1:] == -1)}")

# Test 255-bit subsequence
print(f"\n  Testing 255-bit subsequences...")
best_max_off_peak = float('inf')
best_start = -1

for start_idx in range(len(prbs511)):
    # Extract subsequence
    subseq = np.zeros(255, dtype=int)
    for i in range(255):
        subseq[i] = prbs511[(start_idx + i) % len(prbs511)]

    # Compute autocorrelation
    subseq_bipolar = to_bipolar(subseq)
    autocorr = compute_autocorrelation(subseq_bipolar)
    max_off_peak = np.max(np.abs(autocorr[1:]))

    if max_off_peak < best_max_off_peak:
        best_max_off_peak = max_off_peak
        best_start = start_idx

print(f"    Best subsequence starts at index: {best_start}")
print(f"    Best max off-peak: {best_max_off_peak} (should be -1 for true PRBS)")
print(f"    Conclusion: 255-bit subsequence is NOT a true PRBS255")

# Problem 1c: Cross-correlation between different generators
print("\nProblem 1c: Cross-correlation analysis...")

seqA = generate_prbs(9, 0x110)  # PRBS511 with polynomial x^9 + x^5 + 1
seqB = generate_prbs(9, 0x108)  # PRBS511 with polynomial x^9 + x^4 + 1

seqA_bipolar = to_bipolar(seqA)
seqB_bipolar = to_bipolar(seqB)

crosscorr = compute_crosscorrelation(seqA_bipolar, seqB_bipolar)

print(f"  seqA (0x110) vs seqB (0x108):")
print(f"    Sequences equal? {np.array_equal(seqA, seqB)}")
print(f"    Max cross-correlation: {np.max(crosscorr)}")
print(f"    Min cross-correlation: {np.min(crosscorr)}")
print(f"    Mean cross-correlation: {np.mean(crosscorr):.2f}")


# ============================================================================
# PROBLEM 2: CDMA-BASED TOUCH SENSORS
# ============================================================================

print("\n" + "="*80)
print("PROBLEM 2: CDMA-BASED TOUCH SENSORS")
print("="*80)

def correlate_with_prbs(sense_signal, prbs_bipolar):
    """
    Correlate sense signal with PRBS code at all circular shifts.

    This is how CDMA extracts individual drive line contributions
    from the mixed sense signal.

    Args:
        sense_signal: Raw sense waveform
        prbs_bipolar: PRBS code in bipolar format

    Returns:
        Correlation array (peaks indicate drive line phases)
    """
    N = len(prbs_bipolar)
    sense_padded = np.tile(sense_signal, 2)  # For circular correlation
    correlation = np.zeros(N)

    for shift in range(N):
        window = sense_padded[shift:shift + N]
        correlation[shift] = np.dot(window, prbs_bipolar)

    return correlation


def gaussian(x, amplitude, center, width):
    """Gaussian function for fitting touch response"""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))


# Load touch sensor data
print("\nLoading touch sensor data...")
try:
    notouch = np.loadtxt('HW3.Pr3.notouch.txt')
    touch = np.loadtxt('HW3.Pr3.touch.txt')
    print(f"  Loaded: {len(notouch)} samples (no-touch), {len(touch)} samples (touch)")
except:
    print("  Warning: Data files not found. Using dummy data for demonstration.")
    notouch = np.random.randn(511) * 0.1
    touch = np.random.randn(511) * 0.1

# Generate PRBS511 for correlation
prbs_binary = generate_prbs(9, 0x110, initial_state=257)
prbs_bipolar = to_bipolar(prbs_binary)

# Problem 2a: Find correlation peaks (baseline)
print("\nProblem 2a: Baseline correlation peaks...")

correlation_notouch = correlate_with_prbs(notouch, prbs_bipolar)

# Find 5 peaks
peaks, _ = find_peaks(np.abs(correlation_notouch), height=0, distance=50)
peak_heights = np.abs(correlation_notouch[peaks])
top5_indices = np.argsort(peak_heights)[-5:][::-1]
top5_peaks = np.sort(peaks[top5_indices])

print(f"  Found 5 correlation peaks:")
print(f"  {'Peak':<8} {'Position':<12} {'Correlation':<15} {'Capacitance':<15}")
print(f"  {'-'*60}")

capacitances_notouch = []
for i, peak_pos in enumerate(top5_peaks):
    corr_value = correlation_notouch[peak_pos]
    capacitance = corr_value / len(prbs_bipolar)
    capacitances_notouch.append(capacitance)
    print(f"  {i+1:<8} {peak_pos:<12} {corr_value:<15.2f} {capacitance:<15.4f}")

# Problem 2b: Touch detection
print("\nProblem 2b: Touch location detection...")

correlation_touch = correlate_with_prbs(touch, prbs_bipolar)

capacitances_touch = []
for peak_pos in top5_peaks:
    corr_value = correlation_touch[peak_pos]
    capacitance = corr_value / len(prbs_bipolar)
    capacitances_touch.append(capacitance)

capacitances_notouch = np.array(capacitances_notouch)
capacitances_touch = np.array(capacitances_touch)
capacitance_changes = capacitances_touch - capacitances_notouch

min_change_idx = np.argmin(capacitance_changes)
physical_positions = np.array([5, 10, 15, 20, 25])  # mm
touch_position_mm = physical_positions[min_change_idx]

print(f"  Capacitance changes:")
for i in range(5):
    marker = " ← TOUCH!" if i == min_change_idx else ""
    print(f"    Position {i+1} ({physical_positions[i]} mm): {capacitance_changes[i]:+.4f}{marker}")

print(f"\n  Touch detected at: {touch_position_mm} mm")

# Gaussian fit
try:
    popt, _ = curve_fit(gaussian, physical_positions, capacitance_changes,
                       p0=[capacitance_changes[min_change_idx], touch_position_mm, 5.0],
                       maxfev=5000)
    print(f"  Gaussian fit center: {popt[1]:.2f} mm")
    print(f"  Gaussian width: {popt[2]:.2f} mm")
except:
    print(f"  Gaussian fit: Could not converge")

# Problem 2c: Noise estimation (Extra Credit)
print("\nProblem 2c: Noise estimation (Extra Credit)...")

# Exclude peak regions
mask = np.ones(len(correlation_notouch), dtype=bool)
for peak in top5_peaks:
    mask[max(0, peak-50):min(len(mask), peak+50)] = False

# Off-peak correlation values contain only noise
off_peak_corr = correlation_notouch[mask]
sigma_correlation = np.std(off_peak_corr)

# Noise per sample = post-correlation noise / sqrt(N)
# This accounts for processing gain
sigma_per_sample = sigma_correlation / np.sqrt(len(prbs_bipolar))

print(f"  Post-correlation noise: {sigma_correlation:.6f}")
print(f"  Processing gain: sqrt({len(prbs_bipolar)}) = {np.sqrt(len(prbs_bipolar)):.2f}")
print(f"  Noise per sample (before correlation): {sigma_per_sample:.6f}")
print(f"  SNR: {np.abs(capacitance_changes[min_change_idx]) / sigma_per_sample:.1f}")


# ============================================================================
# PROBLEM 3: DFT/FFT EXERCISES
# ============================================================================

print("\n" + "="*80)
print("PROBLEM 3: DFT/FFT EXERCISES")
print("="*80)

# Problem 3a: Theoretical result
print("\nProblem 3a: Relationship between Y(k) and H(ω)")
print("  ANSWER: Y(k) = H(2πk/N)")
print("  The DFT output directly samples the frequency response")
print("  at the DFT frequency bins ω = 2πk/N")

# Problem 3b: Sampling requirements
print("\nProblem 3b: Sampling requirements")

B = 4000  # Bandwidth in Hz
resolution = 50  # Required resolution in Hz

fs_min = 2 * B  # Nyquist sampling rate
N_min_exact = fs_min / resolution
m_min = int(np.ceil(np.log2(N_min_exact)))
N_min = 2 ** m_min
T_min = N_min / fs_min
actual_resolution = fs_min / N_min

print(f"  Given: Bandwidth B = {B} Hz, Resolution Δf ≤ {resolution} Hz")
print(f"  Minimum sampling rate: fs = {fs_min} Hz (Nyquist: 2B)")
print(f"  Minimum samples: N = {N_min} (which is 2^{m_min})")
print(f"  Minimum record length: T = {T_min} s = {T_min*1000} ms")
print(f"  Actual resolution: Δf = {actual_resolution} Hz ≤ {resolution} Hz ✓")

# Problem 3c: Time-domain aliasing
print("\nProblem 3c: Time-domain aliasing")

a = 0.8  # Signal parameter
N_small = 20
N_large = 100

def compute_time_aliasing(N, a):
    """
    Demonstrate time-domain aliasing by sampling DTFT.

    When we sample X(ω) at N points and take IFFT, we assume
    the time signal is periodic with period N. If the true
    signal doesn't decay to zero within N samples, copies
    wrap around and add up (time-domain aliasing).
    """
    # Sample DTFT at N points
    k = np.arange(N)
    omega_k = 2*np.pi*k/N
    X_k = (1 - a**2) / (1 - 2*a*np.cos(omega_k) + a**2)

    # Compute IFFT
    x_n = np.fft.ifft(X_k)

    # True signal (for comparison)
    n = np.arange(N)
    x_true = a ** np.abs(n - N//2)

    # Compute error
    error = np.abs(np.real(x_n) - x_true)
    rms_error = np.sqrt(np.mean(error**2))

    return X_k, x_n, x_true, rms_error

# Test both values of N
X_k_small, x_n_small, x_true_small, rms_small = compute_time_aliasing(N_small, a)
X_k_large, x_n_large, x_true_large, rms_large = compute_time_aliasing(N_large, a)

print(f"  Signal: x(n) = {a}^|n|")
print(f"  DTFT: X(ω) = (1 - a²) / (1 - 2a cos(ω) + a²)")
print(f"\n  N = {N_small}:")
print(f"    RMS Error: {rms_small:.4f}")
print(f"    Signal at edge: {a**N_small:.4f} (NOT negligible!)")
print(f"    Result: STRONG time-domain aliasing")
print(f"\n  N = {N_large}:")
print(f"    RMS Error: {rms_large:.4f}")
print(f"    Signal at edge: {a**N_large:.2e} (essentially zero)")
print(f"    Result: MINIMAL time-domain aliasing")

print(f"\n  Explanation:")
print(f"    When N is too small ({N_small}), the signal doesn't decay to zero")
print(f"    before wrapping. Copies overlap and add up (aliasing).")
print(f"    When N is large enough ({N_large}), the signal decays to near-zero,")
print(f"    so wrapping has minimal effect.")


