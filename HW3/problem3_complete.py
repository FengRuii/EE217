"""
EE217 Homework 3 - Problem 3 Complete Solution
DFT/FFT Exercises with comprehensive visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


print("="*80)
print("PROBLEM 3: DFT/FFT EXERCISES")
print("="*80)

# ============================================================================
# PROBLEM 3A: Relationship between Y(k) and H(ω)
# ============================================================================

print("\n" + "="*80)
print("PROBLEM 3A: Y(k) vs H(ω) for Periodic Input")
print("="*80)

print("""
Given:
- LTI system with frequency response H(ω)
- Periodic input: x(n) = Σ δ(n - kN) for k = -∞ to ∞
- Output: y(n)
- N-point DFT of y(n): Y(k)

Question: How is Y(k) related to H(ω)?

ANSWER:
Y(k) = H(2πk/N)

Explanation:
1. For an LTI system, output spectrum = input spectrum × frequency response
   Y(ω) = X(ω) · H(ω)

2. For periodic input x(n) with period N, the DTFT is:
   X(ω) = (2π/N) Σ δ(ω - 2πk/N) for k = -∞ to ∞
   (impulses at multiples of 2π/N)

3. Therefore:
   Y(ω) = X(ω) · H(ω) = (2π/N) Σ H(2πk/N)δ(ω - 2πk/N)

4. The N-point DFT samples Y(ω) at ω = 2πk/N:
   Y(k) = Y(ω)|_{ω=2πk/N} = H(2πk/N)

PHYSICAL MEANING:
The DFT output Y(k) directly gives us the frequency response H(ω) evaluated
at the DFT frequency bins ω = 2πk/N. This is because the periodic input
acts like a comb of impulses in the frequency domain, which "samples" the
frequency response at these specific frequencies.

This is the principle behind frequency response measurement!
""")

# Create visualization for 3a
fig_3a, axes = plt.subplots(2, 2, figsize=(14, 10))

N = 16  # Example: 16-point DFT

# Plot 1: Periodic input in time domain
ax = axes[0, 0]
n = np.arange(-N, 2*N)
x_periodic = np.zeros_like(n, dtype=float)
for k in range(-2, 3):
    idx = np.where(n == k*N)[0]
    if len(idx) > 0:
        x_periodic[idx] = 1.0

ax.stem(n, x_periodic, basefmt=' ')
ax.set_xlabel('n (samples)', fontsize=11, fontweight='bold')
ax.set_ylabel('x(n)', fontsize=11, fontweight='bold')
ax.set_title('(a) Periodic Input: x(n) = Σ δ(n - kN)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.set_xlim([-N-5, 2*N+5])

# Plot 2: Frequency spectrum of periodic input
ax = axes[0, 1]
omega = np.linspace(0, 2*np.pi, 1000)
# Draw impulses at 2πk/N
for k in range(N):
    w_k = 2*np.pi*k/N
    ax.plot([w_k, w_k], [0, 1], 'b-', linewidth=2)
    ax.plot(w_k, 1, 'ro', markersize=8)

ax.set_xlabel('ω (radians)', fontsize=11, fontweight='bold')
ax.set_ylabel('|X(ω)| (magnitude)', fontsize=11, fontweight='bold')
ax.set_title('(b) Frequency Spectrum of Periodic Input', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 2*np.pi])
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

# Plot 3: Example frequency response H(ω)
ax = axes[1, 0]
# Example lowpass filter
omega = np.linspace(0, 2*np.pi, 1000)
H_omega = 1 / (1 + 1j*0.5*np.tan((omega - np.pi)/2))
ax.plot(omega, np.abs(H_omega), 'g-', linewidth=2, label='H(ω)')

# Mark DFT frequencies
for k in range(N):
    w_k = 2*np.pi*k/N
    H_k = 1 / (1 + 1j*0.5*np.tan((w_k - np.pi)/2))
    ax.plot(w_k, np.abs(H_k), 'ro', markersize=8)

ax.set_xlabel('ω (radians)', fontsize=11, fontweight='bold')
ax.set_ylabel('|H(ω)|', fontsize=11, fontweight='bold')
ax.set_title('(c) Example Frequency Response H(ω)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 2*np.pi])
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

# Plot 4: DFT output Y(k) = H(2πk/N)
ax = axes[1, 1]
k_vals = np.arange(N)
Y_k = np.zeros(N)
for k in range(N):
    w_k = 2*np.pi*k/N
    H_k = 1 / (1 + 1j*0.5*np.tan((w_k - np.pi)/2))
    Y_k[k] = np.abs(H_k)

ax.stem(k_vals, Y_k, basefmt=' ')
ax.set_xlabel('k (DFT bin)', fontsize=11, fontweight='bold')
ax.set_ylabel('Y(k) = H(2πk/N)', fontsize=11, fontweight='bold')
ax.set_title('(d) DFT Output: Y(k) samples H(ω) at DFT frequencies', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)

fig_3a.suptitle('Problem 3a: Relationship between Y(k) and H(ω)',
                fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('problem3a_relationship.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: problem3a_relationship.png")

# ============================================================================
# PROBLEM 3B: Sampling Requirements
# ============================================================================

print("\n" + "="*80)
print("PROBLEM 3B: Sampling Requirements")
print("="*80)

B = 4000  # Bandwidth in Hz
resolution = 50  # Required resolution in Hz

print(f"\nGiven:")
print(f"  Bandwidth B = {B} Hz")
print(f"  Required resolution Δf ≤ {resolution} Hz")
print(f"  DFT size N = 2^m (power of 2)")

# 1. Minimum sampling rate (Nyquist)
fs_min = 2 * B
print(f"\n1. MINIMUM SAMPLING RATE:")
print(f"   By Nyquist theorem: fs ≥ 2B")
print(f"   fs_min = 2 × {B} Hz = {fs_min} Hz")

# 2. Frequency resolution
print(f"\n2. FREQUENCY RESOLUTION:")
print(f"   Resolution Δf = fs / N = 1 / T")
print(f"   Where T is the record length in seconds")
print(f"   Requirement: Δf ≤ {resolution} Hz")
print(f"   Therefore: N ≥ fs / Δf")

# Minimum N
N_min_exact = fs_min / resolution
print(f"\n   N_min = {fs_min} / {resolution} = {N_min_exact}")

# Round up to next power of 2
m_min = int(np.ceil(np.log2(N_min_exact)))
N_min = 2 ** m_min

print(f"   Since N = 2^m, we need m ≥ {np.log2(N_min_exact):.2f}")
print(f"   Therefore: m = {m_min}, N = 2^{m_min} = {N_min}")

# 3. Record length
T_min = N_min / fs_min
print(f"\n3. MINIMUM RECORD LENGTH:")
print(f"   T = N / fs")
print(f"   T_min = {N_min} / {fs_min} = {T_min} seconds = {T_min * 1000} ms")

# Verify resolution
actual_resolution = fs_min / N_min
print(f"\nVERIFICATION:")
print(f"   Actual resolution: Δf = {fs_min} / {N_min} = {actual_resolution} Hz")
print(f"   Required: Δf ≤ {resolution} Hz")
print(f"   ✓ Satisfies requirement: {actual_resolution} ≤ {resolution}")

print(f"\n{'='*80}")
print(f"ANSWERS:")
print(f"  Minimum sampling rate: {fs_min} Hz")
print(f"  Minimum number of samples: N = {N_min} (which is 2^{m_min})")
print(f"  Minimum record length: T = {T_min} s = {T_min * 1000} ms")
print(f"{'='*80}")

# Create visualization for 3b
fig_3b, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Frequency resolution concept
ax = axes[0, 0]
freq_bins = np.arange(0, fs_min/2 + 1, fs_min/N_min)
ax.stem(freq_bins[:20], np.ones(min(20, len(freq_bins))), basefmt=' ')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
ax.set_ylabel('DFT bin', fontsize=11, fontweight='bold')
ax.set_title(f'(a) Frequency Resolution: Δf = {actual_resolution} Hz',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add resolution annotation
ax.annotate('', xy=(fs_min/N_min, 0.5), xytext=(0, 0.5),
            arrowprops=dict(arrowstyle='<->', lw=2, color='red'))
ax.text(fs_min/(2*N_min), 0.6, f'Δf = {actual_resolution} Hz',
        ha='center', fontsize=10, fontweight='bold', color='red',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Plot 2: Nyquist criterion
ax = axes[0, 1]
freqs = np.linspace(0, 2*B, 1000)
# Analog signal spectrum (rectangular, bandlimited)
spectrum = np.zeros_like(freqs)
spectrum[freqs <= B] = 1.0

ax.fill_between(freqs, 0, spectrum, alpha=0.3, color='blue', label='Signal spectrum')
ax.axvline(x=B, color='b', linestyle='--', linewidth=2, label=f'Bandwidth B = {B} Hz')
ax.axvline(x=fs_min/2, color='r', linestyle='--', linewidth=2,
           label=f'Nyquist freq = fs/2 = {fs_min/2} Hz')

ax.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
ax.set_ylabel('Magnitude', fontsize=11, fontweight='bold')
ax.set_title('(b) Nyquist Sampling Criterion', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 2*B])

# Plot 3: Time-domain sampling
ax = axes[1, 0]
t_analog = np.linspace(0, 0.01, 1000)
x_analog = np.sin(2*np.pi*1000*t_analog)  # 1 kHz signal

ax.plot(t_analog, x_analog, 'b-', linewidth=1, alpha=0.5, label='Analog signal')

# Sampled points
t_sampled = np.arange(0, 0.01, 1/fs_min)
x_sampled = np.sin(2*np.pi*1000*t_sampled)
ax.stem(t_sampled, x_sampled, linefmt='r-', markerfmt='ro', basefmt=' ',
        label=f'Sampled at fs = {fs_min} Hz')

ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
ax.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
ax.set_title('(c) Time-Domain Sampling', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 4: Trade-off diagram
ax = axes[1, 1]
N_values = 2 ** np.arange(6, 11)  # 64 to 1024
resolutions = fs_min / N_values
record_lengths = N_values / fs_min

ax.plot(N_values, resolutions, 'bo-', linewidth=2, markersize=8, label='Resolution (Hz)')
ax.axhline(y=resolution, color='r', linestyle='--', linewidth=2,
           label=f'Required: {resolution} Hz')
ax.axvline(x=N_min, color='g', linestyle='--', linewidth=2,
           label=f'Chosen N = {N_min}')

ax.set_xlabel('N (number of samples)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency Resolution (Hz)', fontsize=11, fontweight='bold')
ax.set_title('(d) Resolution vs Sample Count', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log', base=2)

# Annotate the chosen point
ax.plot(N_min, actual_resolution, 'go', markersize=15, markeredgewidth=3,
        markerfacecolor='none')

fig_3b.suptitle('Problem 3b: Sampling Requirements for DFT Analysis',
                fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('problem3b_sampling.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: problem3b_sampling.png")

# ============================================================================
# PROBLEM 3C: Time-Domain Aliasing
# ============================================================================

print("\n" + "="*80)
print("PROBLEM 3C: Time-Domain Aliasing")
print("="*80)

a = 0.8  # Given parameter
print(f"\nGiven:")
print(f"  Signal: x(n) = a^|n|, -1 < a < 1")
print(f"  Parameter: a = {a}")
print(f"  DTFT: X(ω) = (1 - a²) / (1 - 2a cos(ω) + a²)")

# Compute DTFT
omega = np.linspace(0, 2*np.pi, 1000)
X_omega = (1 - a**2) / (1 - 2*a*np.cos(omega) + a**2)

print(f"\n1. Plotting DTFT X(ω) for 0 ≤ ω ≤ 2π")
print(f"   Maximum value: {np.max(X_omega):.4f} at ω = 0")
print(f"   Minimum value: {np.min(X_omega):.4f} at ω = π")

# Test two different N values
N_small = 20
N_large = 100

print(f"\n2. Testing with N = {N_small} and N = {N_large}")

# Function to compute time-domain signal from frequency samples
def compute_time_domain(N, a):
    # Sample DTFT at N points
    k = np.arange(N)
    omega_k = 2*np.pi*k/N
    X_k = (1 - a**2) / (1 - 2*a*np.cos(omega_k) + a**2)

    # Compute IFFT
    x_n = np.fft.ifft(X_k)

    # True signal
    n = np.arange(N)
    x_true = a ** np.abs(n - N//2)  # Centered version

    # For comparison, generate longer true signal
    n_long = np.arange(-N, N)
    x_true_long = a ** np.abs(n_long)

    return X_k, x_n, x_true, n_long, x_true_long

# Compute for both N values
X_k_small, x_n_small, x_true_small, n_long_small, x_true_long_small = compute_time_domain(N_small, a)
X_k_large, x_n_large, x_true_large, n_long_large, x_true_long_large = compute_time_domain(N_large, a)

print(f"\nResults for N = {N_small}:")
print(f"  IFFT is real? {np.max(np.abs(x_n_small.imag)) < 1e-10}")
print(f"  Signal appears periodic in time domain")

print(f"\nResults for N = {N_large}:")
print(f"  IFFT is real? {np.max(np.abs(x_n_large.imag)) < 1e-10}")
print(f"  Signal much closer to true exponential decay")

print(f"\n3. What's happening when N = {N_small}?")
print(f"   TIME-DOMAIN ALIASING!")
print(f"   - The true signal x(n) = a^|n| extends from -∞ to +∞")
print(f"   - When we sample X(ω) at only N = {N_small} points, we're assuming")
print(f"     the time-domain signal is periodic with period N")
print(f"   - This causes ALIASING: copies of the signal wrap around and add up")
print(f"   - The IFFT gives us the ALIASED sum, not the true signal")
print(f"   - With N = {N_large}, we sample X(ω) more densely, so the signal")
print(f"     decays to near-zero before wrapping, reducing aliasing")

# Create visualization for 3c
fig_3c = plt.figure(figsize=(16, 12))

# Plot 1: DTFT X(ω)
ax1 = plt.subplot(3, 3, 1)
ax1.plot(omega, X_omega, 'b-', linewidth=2)
ax1.set_xlabel('ω (radians)', fontsize=10, fontweight='bold')
ax1.set_ylabel('X(ω)', fontsize=10, fontweight='bold')
ax1.set_title('(a) DTFT: X(ω)', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 2*np.pi])
ax1.set_xticks([0, np.pi, 2*np.pi])
ax1.set_xticklabels(['0', 'π', '2π'])

# Plot 2: Frequency samples for N=20
ax2 = plt.subplot(3, 3, 2)
k_small = np.arange(N_small)
omega_k_small = 2*np.pi*k_small/N_small
ax2.plot(omega, X_omega, 'b-', linewidth=1, alpha=0.3, label='X(ω)')
ax2.stem(omega_k_small, X_k_small, linefmt='r-', markerfmt='ro', basefmt=' ',
         label=f'X(2πk/{N_small})')
ax2.set_xlabel('ω (radians)', fontsize=10, fontweight='bold')
ax2.set_ylabel('X(ω)', fontsize=10, fontweight='bold')
ax2.set_title(f'(b) Frequency Samples: N = {N_small}', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 2*np.pi])
ax2.set_xticks([0, np.pi, 2*np.pi])
ax2.set_xticklabels(['0', 'π', '2π'])

# Plot 3: Frequency samples for N=100
ax3 = plt.subplot(3, 3, 3)
k_large = np.arange(N_large)
omega_k_large = 2*np.pi*k_large/N_large
ax3.plot(omega, X_omega, 'b-', linewidth=1, alpha=0.3, label='X(ω)')
ax3.stem(omega_k_large[::5], X_k_large[::5], linefmt='r-', markerfmt='ro', basefmt=' ',
         label=f'X(2πk/{N_large})')
ax3.set_xlabel('ω (radians)', fontsize=10, fontweight='bold')
ax3.set_ylabel('X(ω)', fontsize=10, fontweight='bold')
ax3.set_title(f'(c) Frequency Samples: N = {N_large}', fontsize=11, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0, 2*np.pi])
ax3.set_xticks([0, np.pi, 2*np.pi])
ax3.set_xticklabels(['0', 'π', '2π'])

# Plot 4: True signal (extended)
ax4 = plt.subplot(3, 3, 4)
ax4.stem(n_long_small, x_true_long_small, linefmt='b-', markerfmt='bo',
         basefmt=' ', label='True x(n) = a^|n|')
ax4.axvline(x=0, color='r', linestyle='--', linewidth=1)
ax4.axvline(x=N_small, color='r', linestyle='--', linewidth=1)
ax4.set_xlabel('n', fontsize=10, fontweight='bold')
ax4.set_ylabel('x(n)', fontsize=10, fontweight='bold')
ax4.set_title(f'(d) True Signal (extends to ±∞)', fontsize=11, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_xlim([-N_small, 2*N_small])

# Plot 5: IFFT result for N=20
ax5 = plt.subplot(3, 3, 5)
n_small = np.arange(N_small)
ax5.stem(n_small, np.real(x_n_small), linefmt='r-', markerfmt='ro',
         basefmt=' ', label=f'IFFT (N={N_small})')
ax5.stem(n_small, x_true_small, linefmt='b-', markerfmt='bx',
         basefmt=' ', label='True (one period)')
ax5.set_xlabel('n', fontsize=10, fontweight='bold')
ax5.set_ylabel('x(n)', fontsize=10, fontweight='bold')
ax5.set_title(f'(e) IFFT Result: N = {N_small} (ALIASED!)', fontsize=11, fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Plot 6: IFFT result for N=100
ax6 = plt.subplot(3, 3, 6)
n_large = np.arange(N_large)
ax6.stem(n_large[::5], np.real(x_n_large)[::5], linefmt='r-', markerfmt='ro',
         basefmt=' ', label=f'IFFT (N={N_large})')
ax6.stem(n_large[::5], x_true_large[::5], linefmt='b-', markerfmt='bx',
         basefmt=' ', label='True (one period)')
ax6.set_xlabel('n', fontsize=10, fontweight='bold')
ax6.set_ylabel('x(n)', fontsize=10, fontweight='bold')
ax6.set_title(f'(f) IFFT Result: N = {N_large} (Much Better!)', fontsize=11, fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# Plot 7: Aliasing illustration for N=20
ax7 = plt.subplot(3, 3, 7)
# Show how signal wraps around
for shift in [-N_small, 0, N_small]:
    x_shifted = a ** np.abs(n_small - shift)
    ax7.plot(n_small, x_shifted, alpha=0.3, label=f'Shift by {shift}')
ax7.plot(n_small, np.real(x_n_small), 'r-', linewidth=3, label='IFFT (sum of all)')
ax7.set_xlabel('n', fontsize=10, fontweight='bold')
ax7.set_ylabel('x(n)', fontsize=10, fontweight='bold')
ax7.set_title(f'(g) Aliasing Mechanism (N={N_small})', fontsize=11, fontweight='bold')
ax7.legend(fontsize=7)
ax7.grid(True, alpha=0.3)

# Plot 8: Error analysis for N=20
ax8 = plt.subplot(3, 3, 8)
error_small = np.real(x_n_small) - x_true_small
ax8.stem(n_small, error_small, linefmt='r-', markerfmt='ro', basefmt=' ')
ax8.axhline(y=0, color='k', linewidth=0.5)
ax8.set_xlabel('n', fontsize=10, fontweight='bold')
ax8.set_ylabel('Error', fontsize=10, fontweight='bold')
ax8.set_title(f'(h) Reconstruction Error (N={N_small})', fontsize=11, fontweight='bold')
ax8.grid(True, alpha=0.3)
rms_error_small = np.sqrt(np.mean(error_small**2))
ax8.text(0.5, 0.95, f'RMS Error: {rms_error_small:.4f}',
         transform=ax8.transAxes, ha='center', va='top', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Plot 9: Error analysis for N=100
ax9 = plt.subplot(3, 3, 9)
error_large = np.real(x_n_large) - x_true_large
ax9.stem(n_large[::5], error_large[::5], linefmt='r-', markerfmt='ro', basefmt=' ')
ax9.axhline(y=0, color='k', linewidth=0.5)
ax9.set_xlabel('n', fontsize=10, fontweight='bold')
ax9.set_ylabel('Error', fontsize=10, fontweight='bold')
ax9.set_title(f'(i) Reconstruction Error (N={N_large})', fontsize=11, fontweight='bold')
ax9.grid(True, alpha=0.3)
rms_error_large = np.sqrt(np.mean(error_large**2))
ax9.text(0.5, 0.95, f'RMS Error: {rms_error_large:.4f}',
         transform=ax9.transAxes, ha='center', va='top', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

fig_3c.suptitle('Problem 3c: Time-Domain Aliasing Demonstration',
                fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('problem3c_aliasing.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: problem3c_aliasing.png")

# Create simplified comparison plot
fig_3c_simple, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: N=20
ax = axes[0]
ax.plot(n_long_small, x_true_long_small, 'b-', linewidth=2, alpha=0.5,
        label='True signal (infinite extent)')
ax.stem(n_small, np.real(x_n_small), linefmt='r-', markerfmt='ro',
        basefmt=' ', label=f'IFFT result (N={N_small})')

# Show periodicity
for shift in [-N_small, N_small]:
    ax.axvline(x=shift, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax.axvspan(0, N_small, alpha=0.1, color='yellow', label='One period')

ax.set_xlabel('n (samples)', fontsize=12, fontweight='bold')
ax.set_ylabel('x(n)', fontsize=12, fontweight='bold')
ax.set_title(f'N = {N_small}: Strong Time-Domain Aliasing', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([-N_small, 2*N_small])

# Add text box
textstr = f'RMS Error: {rms_error_small:.4f}\nSignal tails overlap!\nCauses distortion.'
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.8))

# Right: N=100
ax = axes[1]
ax.plot(n_long_large, x_true_long_large, 'b-', linewidth=2, alpha=0.5,
        label='True signal (infinite extent)')
ax.stem(n_large[::5], np.real(x_n_large)[::5], linefmt='r-', markerfmt='ro',
        basefmt=' ', label=f'IFFT result (N={N_large})')

# Show periodicity
for shift in [-N_large, N_large]:
    ax.axvline(x=shift, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax.axvspan(0, N_large, alpha=0.1, color='yellow', label='One period')

ax.set_xlabel('n (samples)', fontsize=12, fontweight='bold')
ax.set_ylabel('x(n)', fontsize=12, fontweight='bold')
ax.set_title(f'N = {N_large}: Minimal Time-Domain Aliasing', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([-N_large, 2*N_large])

# Add text box
textstr = f'RMS Error: {rms_error_large:.4f}\nSignal decays to ~0\nbefore wrapping.\nMinimal aliasing!'
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('problem3c_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: problem3c_comparison.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("PROBLEM 3 COMPLETE - SUMMARY")
print("="*80)

print(f"\nProblem 3a: Y(k) = H(2πk/N)")
print(f"  The DFT directly samples the frequency response")

print(f"\nProblem 3b: Sampling Requirements")
print(f"  Minimum sampling rate: {fs_min} Hz")
print(f"  Minimum samples: N = {N_min} (2^{m_min})")
print(f"  Minimum record length: {T_min} seconds")

print(f"\nProblem 3c: Time-Domain Aliasing")
print(f"  N = {N_small}: Strong aliasing (RMS error = {rms_error_small:.4f})")
print(f"  N = {N_large}: Minimal aliasing (RMS error = {rms_error_large:.4f})")
print(f"  Cause: Sampling frequency domain too coarsely assumes periodic time signal")

print("\n" + "="*80)
print("ALL PLOTS GENERATED")
print("="*80)
print("\nFiles created:")
print("  - problem3a_relationship.png (4 subplots)")
print("  - problem3b_sampling.png (4 subplots)")
print("  - problem3c_aliasing.png (9 subplots)")
print("  - problem3c_comparison.png (2 subplots)")
print()
