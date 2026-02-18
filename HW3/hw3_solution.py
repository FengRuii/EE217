"""
EE217 Homework 3 Solution
Winter 2026

This script solves all three problems:
- Problem 1: PRBS Signaling
- Problem 2: CDMA-Based Touch Sensors
- Problem 3: DFT/FFT Exercises
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# ============================================================================
# PROBLEM 1: PRBS SIGNALING
# ============================================================================

def generate_prbs(n_bits, polynomial_hex, initial_state=1):
    """
    Generate a PRBS (Pseudo-Random Binary Sequence) using a Galois LFSR.

    Args:
        n_bits: Number of bits in the shift register (e.g., 7 for PRBS7)
        polynomial_hex: Generator polynomial in hex format (e.g., 0x60 for PRBS7)
        initial_state: Initial state of the shift register (default=1)

    Returns:
        numpy array of the PRBS sequence (0s and 1s)

    How it works:
        Galois LFSR implementation:
        1. Output the current LSB
        2. Shift right
        3. If the output bit was 1, XOR with the polynomial
    """
    # The period of a maximal-length PRBS is 2^n - 1
    period = (1 << n_bits) - 1  # Same as 2**n_bits - 1

    # Initialize the shift register with the initial state
    state = initial_state

    # Array to store the output sequence
    sequence = np.zeros(period, dtype=int)

    # Generate the sequence using Galois LFSR
    for i in range(period):
        # Output the LSB
        output_bit = state & 1
        sequence[i] = output_bit

        # Shift right
        state >>= 1

        # If output bit was 1, XOR with polynomial
        if output_bit:
            state ^= polynomial_hex

    return sequence


def prbs_to_bipolar(sequence):
    """
    Convert binary sequence (0, 1) to bipolar sequence (-1, +1).
    This is needed for correlation analysis.

    Args:
        sequence: Binary sequence (0s and 1s)

    Returns:
        Bipolar sequence (-1s and +1s)
    """
    return 2 * sequence - 1


def compute_autocorrelation(sequence):
    """
    Compute the autocorrelation of a PRBS sequence.

    For a good PRBS, we expect:
    - A peak of height N at zero offset
    - Value of -1 at all other offsets

    Args:
        sequence: Bipolar sequence (-1, +1)

    Returns:
        Autocorrelation array
    """
    N = len(sequence)
    autocorr = np.zeros(N)

    for offset in range(N):
        # Circular correlation: compare sequence with shifted version of itself
        autocorr[offset] = np.sum(sequence * np.roll(sequence, offset))

    return autocorr


# Problem 1a: Generate PRBS sequences
print("=" * 70)
print("PROBLEM 1a: Generate PRBS sequences")
print("=" * 70)

# Define the polynomials from the table
prbs_configs = {
    'PRBS7': {'n_bits': 7, 'polynomial': 0x60, 'period': 127},
    'PRBS127': {'n_bits': 7, 'polynomial': 0x60, 'period': 127},  # Same as PRBS7
    'PRBS511': {'n_bits': 9, 'polynomial': 0x110, 'period': 511},
    'PRBS1023': {'n_bits': 10, 'polynomial': 0x240, 'period': 1023}
}

# Generate all sequences
prbs_sequences = {}
for name, config in prbs_configs.items():
    seq = generate_prbs(config['n_bits'], config['polynomial'])
    prbs_sequences[name] = seq
    print(f"{name}: Generated {len(seq)} bits")
    print(f"  First 20 bits: {seq[:20]}")
    print()


# Problem 1b: Autocorrelation of PRBS511
print("=" * 70)
print("PROBLEM 1b: Autocorrelation of PRBS511")
print("=" * 70)

# Convert PRBS511 to bipolar
prbs511_bipolar = prbs_to_bipolar(prbs_sequences['PRBS511'])

# Compute autocorrelation
autocorr_511 = compute_autocorrelation(prbs511_bipolar)

print(f"Autocorrelation at zero offset: {autocorr_511[0]}")
print(f"Autocorrelation at offset 1: {autocorr_511[1]}")
print(f"Expected: Peak of 511 at zero, -1 elsewhere")
print()

# Plot autocorrelation
plt.figure(figsize=(12, 4))
plt.plot(autocorr_511)
plt.xlabel('Offset (samples)')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of PRBS511')
plt.grid(True)
plt.axhline(y=-1, color='r', linestyle='--', alpha=0.5, label='Expected off-peak value (-1)')
plt.legend()
plt.tight_layout()
plt.savefig('problem1b_autocorr_prbs511.png', dpi=150)
print("Saved: problem1b_autocorr_prbs511.png")
print()

# Check if 255-bit subsequence is pseudorandom
print("Checking 255-bit subsequences...")
best_subsequence_start = 0
best_max_offpeak = float('inf')

# Try all possible 255-bit subsequences
for start_idx in range(511):
    subseq = np.roll(prbs511_bipolar, -start_idx)[:255]
    autocorr_sub = compute_autocorrelation(subseq)

    # Find maximum off-peak autocorrelation
    max_offpeak = np.max(np.abs(autocorr_sub[1:]))

    if max_offpeak < best_max_offpeak:
        best_max_offpeak = max_offpeak
        best_subsequence_start = start_idx

print(f"Best 255-bit subsequence starts at index: {best_subsequence_start}")
print(f"Maximum off-peak autocorrelation: {best_max_offpeak}")
print(f"Expected for true PRBS255: -1")
print()

# Plot the best subsequence autocorrelation
best_subseq = np.roll(prbs511_bipolar, -best_subsequence_start)[:255]
autocorr_best_sub = compute_autocorrelation(best_subseq)

plt.figure(figsize=(12, 4))
plt.plot(autocorr_best_sub)
plt.xlabel('Offset (samples)')
plt.ylabel('Autocorrelation')
plt.title(f'Autocorrelation of Best 255-bit Subsequence (starting at index {best_subsequence_start})')
plt.grid(True)
plt.axhline(y=-1, color='r', linestyle='--', alpha=0.5, label='Expected off-peak value (-1)')
plt.legend()
plt.tight_layout()
plt.savefig('problem1b_autocorr_255bit.png', dpi=150)
print("Saved: problem1b_autocorr_255bit.png")
print()

print("Answer: A 255-bit subsequence of PRBS511 is NOT a true PRBS255.")
print("Reason: The off-peak autocorrelation values are not all -1.")
print("It DOES depend on which subsequence you choose - some are better than others.")
print()


# Problem 1c: Cross-correlation between two PRBS511 generators
print("=" * 70)
print("PROBLEM 1c: Cross-correlation of two PRBS511 sequences")
print("=" * 70)

# Generate seqA (polynomial 0x110) - already have this
seqA_bipolar = prbs511_bipolar

# Generate seqB (polynomial 0x108)
seqB = generate_prbs(9, 0x108)
seqB_bipolar = prbs_to_bipolar(seqB)

# Check if they're equal
if np.array_equal(seqA_bipolar, seqB_bipolar):
    print("seqA equals seqB: YES")
else:
    print("seqA equals seqB: NO")
print()

# Compute cross-correlation
N = 511
xcorr = np.zeros(N)
for n in range(N):
    xcorr[n] = np.sum(seqA_bipolar * np.roll(seqB_bipolar, n))

max_xcorr = np.max(np.abs(xcorr))
print(f"Maximum cross-correlation: {max_xcorr}")
print()

# Plot cross-correlation
plt.figure(figsize=(12, 4))
plt.plot(xcorr)
plt.xlabel('Time offset (samples)')
plt.ylabel('Cross-correlation')
plt.title('Cross-correlation between seqA (0x110) and seqB (0x108)')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('problem1c_crosscorr.png', dpi=150)
print("Saved: problem1c_crosscorr.png")
print()


# ============================================================================
# PROBLEM 2: CDMA-BASED TOUCH SENSORS
# ============================================================================

print("=" * 70)
print("PROBLEM 2: CDMA-Based Touch Sensors")
print("=" * 70)

# Problem 2a: Find correlation peaks for notouch data
print("\nProblem 2a: Analyzing notouch sensor data")
print("-" * 70)

# Load the sense waveform data
notouch_data = np.loadtxt('HW3.Pr3.notouch.txt')
print(f"Loaded notouch data: {len(notouch_data)} samples")

# Generate PRBS511 with initial state 257 (as specified in problem)
# Initial state 257 = 0b100000001 in binary
prbs511_257 = generate_prbs(9, 0x110, initial_state=257)
prbs511_257_bipolar = prbs_to_bipolar(prbs511_257)

print(f"PRBS511 sequence length: {len(prbs511_257)}")
print(f"Initial state: 257 (binary: {bin(257)})")
print()

# Correlate the sense data with the PRBS sequence
correlation = np.correlate(notouch_data, prbs511_257_bipolar, mode='full')

# The correlation will be longer than the original signal
# We only need the part corresponding to valid overlap
correlation = correlation[len(prbs511_257_bipolar)-1:]

# Find the peaks in the correlation
# There should be 5 peaks corresponding to the 5 drive lines
from scipy.signal import find_peaks

# Find all peaks with some minimum prominence
all_peaks, properties = find_peaks(correlation, distance=50, prominence=5)

# Select the 5 strongest peaks (highest correlation values)
peak_heights = correlation[all_peaks]
top5_indices = np.argsort(peak_heights)[-5:]  # Get indices of 5 largest
peaks = all_peaks[top5_indices]
peak_values = correlation[peaks]

# Sort by time offset (phase)
sorted_indices = np.argsort(peaks)
peaks = peaks[sorted_indices]
peak_values = peak_values[sorted_indices]

print(f"Found {len(peaks)} strongest correlation peaks (out of {len(all_peaks)} total)")
print("\nDrive/Sense Pair Information:")
print("Peak #  | Phase Offset (samples) | Capacitance (relative)")
print("-" * 60)
for i, (offset, cap) in enumerate(zip(peaks, peak_values), 1):
    print(f"  {i}     |        {offset:4d}            |      {cap:8.4f}")
print()


# Problem 2b: Find touch location
print("\nProblem 2b: Finding touch location")
print("-" * 70)

# Load the touch data
touch_data = np.loadtxt('HW3.Pr3.touch.txt')
print(f"Loaded touch data: {len(touch_data)} samples")

# Correlate with PRBS again
correlation_touch = np.correlate(touch_data, prbs511_257_bipolar, mode='full')
correlation_touch = correlation_touch[len(prbs511_257_bipolar)-1:]

# Extract capacitance values at the same peak positions
touch_capacitances = correlation_touch[peaks]

# Calculate the difference (touch signal)
cap_difference = touch_capacitances - peak_values

print("\nCapacitance changes due to touch:")
print("Position | Notouch Cap | Touch Cap | Difference")
print("-" * 60)
positions_mm = [5, 10, 15, 20, 25]
for i, (pos, no, yes, diff) in enumerate(zip(positions_mm, peak_values,
                                               touch_capacitances, cap_difference)):
    print(f"  {pos} mm  |   {no:8.4f}  | {yes:8.4f} |  {diff:8.4f}")
print()

# Fit a Gaussian to find the touch location
# The touch response should be Gaussian in space
from scipy.optimize import curve_fit

def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2))

# Fit Gaussian to the capacitance differences
popt, _ = curve_fit(gaussian, positions_mm, cap_difference,
                    p0=[np.max(cap_difference), 15, 5])

touch_location = popt[1]
print(f"Estimated touch location: {touch_location:.2f} mm")
print(f"Touch amplitude: {popt[0]:.4f}")
print(f"Touch spread (sigma): {popt[2]:.2f} mm")
print()

# Plot the results
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Correlation functions
axes[0].plot(correlation[:1000], label='No touch', alpha=0.7)
axes[0].plot(correlation_touch[:1000], label='With touch', alpha=0.7)
axes[0].plot(peaks[peaks < 1000], correlation[peaks[peaks < 1000]], 'ro',
             label='Detected peaks', markersize=8)
axes[0].set_xlabel('Sample offset')
axes[0].set_ylabel('Correlation value')
axes[0].set_title('Correlation of Sense Data with PRBS511')
axes[0].legend()
axes[0].grid(True)

# Plot 2: Touch location
axes[1].plot(positions_mm, cap_difference, 'bo-', label='Measured', markersize=10)
x_fit = np.linspace(5, 25, 100)
y_fit = gaussian(x_fit, *popt)
axes[1].plot(x_fit, y_fit, 'r-', label='Gaussian fit')
axes[1].axvline(touch_location, color='g', linestyle='--',
                label=f'Touch location: {touch_location:.2f} mm')
axes[1].set_xlabel('Position (mm)')
axes[1].set_ylabel('Capacitance change')
axes[1].set_title('Touch Location Estimation')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('problem2_touch_sensor.png', dpi=150)
print("Saved: problem2_touch_sensor.png")
print()


# Problem 2c: Estimate noise
print("\nProblem 2c: Noise estimation")
print("-" * 70)

# The noise in each sample can be estimated by looking at the residual
# after removing the expected signal
# One approach: use the correlation to reconstruct the expected signal,
# then compute the difference

# For simplicity, we can estimate noise from regions where there's no signal
# Or we can use the standard deviation of the off-peak correlation values

# Method: Look at the correlation values away from the peaks
mask = np.ones(len(correlation), dtype=bool)
for peak in peaks:
    # Exclude ±50 samples around each peak
    mask[max(0, peak-50):min(len(mask), peak+50)] = False

off_peak_corr = correlation[mask]
noise_std = np.std(off_peak_corr)

print(f"Estimated noise standard deviation: {noise_std:.6f}")
print(f"This represents the noise per sample before correlation.")
print()

# The noise per sample before correlation is related to the noise after
# correlation by the processing gain (length of PRBS sequence)
noise_per_sample = noise_std / np.sqrt(len(prbs511_257_bipolar))
print(f"Estimated noise per sample (before correlation): {noise_per_sample:.6f}")
print()


# ============================================================================
# PROBLEM 3: DFT/FFT EXERCISES
# ============================================================================

print("=" * 70)
print("PROBLEM 3: DFT/FFT Exercises")
print("=" * 70)

# Problem 3a: Theoretical question
print("\nProblem 3a: Relationship between Y(k) and H(ω)")
print("-" * 70)
print("For a periodic input x(n) with period N, the output y(n) of an LTI system")
print("with frequency response H(ω) has the following relationship:")
print()
print("  Y(k) = H(2πk/N) * X(k)")
print()
print("where X(k) is the N-point DFT of the input x(n).")
print()
print("Explanation:")
print("- The periodic input can be decomposed into frequency components at ω = 2πk/N")
print("- Each frequency component is scaled by the system's frequency response H(ω)")
print("- The DFT Y(k) represents samples of the frequency response at these discrete points")
print()


# Problem 3b: Minimum sampling requirements
print("\nProblem 3b: Sampling requirements for DFT")
print("-" * 70)

B = 4000  # Bandwidth in Hz
resolution = 50  # Desired resolution in Hz

# Minimum sampling rate (Nyquist criterion)
fs_min = 2 * B
print(f"Bandwidth B = {B} Hz")
print(f"Desired frequency resolution = {resolution} Hz")
print()
print(f"Minimum sampling rate (Nyquist): fs_min = 2B = {fs_min} Hz")
print()

# Minimum record length
T_min = 1 / resolution
print(f"Minimum record length: T_min = 1/Δf = {T_min} s")
print()

# Minimum number of samples
N_min = fs_min * T_min
print(f"Minimum number of samples: N_min = fs_min * T_min = {N_min}")
print()

# Round up to nearest power of 2
N_fft = int(2**np.ceil(np.log2(N_min)))
print(f"Since N = 2^m, we need: N = {N_fft} samples")
print(f"This is 2^{int(np.log2(N_fft))}")
print()


# Problem 3c: Frequency sampling and time-domain aliasing
print("\nProblem 3c: Inverse FFT of frequency samples")
print("-" * 70)

a = 0.8
print(f"Signal: x(n) = {a}^|n|, where a = {a}")
print()

# Define the DTFT
def X_omega(omega, a):
    return (1 - a**2) / (1 - 2*a*np.cos(omega) + a**2)

# Plot X(ω) for 0 ≤ ω ≤ 2π
omega = np.linspace(0, 2*np.pi, 1000)
X_vals = X_omega(omega, a)

plt.figure(figsize=(12, 4))
plt.plot(omega, X_vals, 'b-', linewidth=2)
plt.xlabel('ω (radians)')
plt.ylabel('X(ω)')
plt.title(f'DTFT of x(n) = {a}^|n|')
plt.grid(True)
plt.tight_layout()
plt.savefig('problem3c_dtft.png', dpi=150)
print("Saved: problem3c_dtft.png")
print()

# Generate time-domain signals using IFFT for N=20 and N=100
N_values = [20, 100]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for idx, N in enumerate(N_values):
    print(f"Processing N = {N}...")

    # Sample the frequency domain
    k = np.arange(N)
    omega_k = 2 * np.pi * k / N
    X_k = X_omega(omega_k, a)

    # Compute IFFT to get time-domain signal
    x_n_reconstructed = np.fft.ifft(X_k).real

    # True signal for comparison (compute for a longer range to show aliasing)
    n = np.arange(-50, 51)
    x_n_true = a**np.abs(n)

    # Plot frequency domain
    axes[idx, 0].stem(k, X_k, basefmt=' ')
    axes[idx, 0].plot(omega*N/(2*np.pi), X_vals, 'r-', alpha=0.3, label='True DTFT')
    axes[idx, 0].set_xlabel('k (frequency bin)')
    axes[idx, 0].set_ylabel('X(k)')
    axes[idx, 0].set_title(f'Frequency Domain (N={N})')
    axes[idx, 0].grid(True)
    axes[idx, 0].legend()

    # Plot time domain
    axes[idx, 1].stem(np.arange(N), x_n_reconstructed, basefmt=' ', label='IFFT result')

    # Overlay the true signal (shifted to show periodic extension)
    n_overlay = np.arange(-10, N+10)
    x_overlay = a**np.abs(n_overlay)
    axes[idx, 1].plot(n_overlay, x_overlay, 'r-', alpha=0.5, label='True signal')

    axes[idx, 1].set_xlabel('n (sample)')
    axes[idx, 1].set_ylabel('x(n)')
    axes[idx, 1].set_title(f'Time Domain Reconstruction (N={N})')
    axes[idx, 1].grid(True)
    axes[idx, 1].legend()
    axes[idx, 1].set_xlim(-5, N+5)

    print(f"  First 10 samples: {x_n_reconstructed[:10]}")
    print()

plt.tight_layout()
plt.savefig('problem3c_ifft_comparison.png', dpi=150)
print("Saved: problem3c_ifft_comparison.png")
print()

print("What's happening when N=20?")
print("-" * 70)
print("TIME-DOMAIN ALIASING (or circular time-aliasing)!")
print()
print("The true signal x(n) = 0.8^|n| extends infinitely in both directions.")
print("When we sample the DTFT at N discrete points and take the IFFT, we get")
print("a PERIODIC time-domain signal with period N.")
print()
print("For N=20:")
print("  - The signal tail at n > 10 wraps around and appears at n < 0")
print("  - The signal tail at n < -10 wraps around and appears at n > 0")
print("  - This aliasing causes the reconstructed signal to differ from the true signal")
print()
print("For N=100:")
print("  - The period is much longer, so the tails have decayed significantly")
print("  - Less aliasing occurs, and the reconstruction is more accurate near n=0")
print()
print("This is analogous to frequency-domain aliasing in sampling, but occurring")
print("in the time domain due to frequency-domain sampling!")
print()

print("=" * 70)
print("ALL PROBLEMS COMPLETED!")
print("=" * 70)
print("\nGenerated files:")
print("  - problem1b_autocorr_prbs511.png")
print("  - problem1b_autocorr_255bit.png")
print("  - problem1c_crosscorr.png")
print("  - problem2_touch_sensor.png")
print("  - problem3c_dtft.png")
print("  - problem3c_ifft_comparison.png")
