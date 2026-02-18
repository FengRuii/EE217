"""
Problem 2: Comprehensive Additional Plots
Creates detailed visualizations for every aspect of the touch sensor analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def generate_prbs(n_bits, polynomial_hex, initial_state=1):
    """Generate PRBS using Galois LFSR"""
    period = (1 << n_bits) - 1
    state = initial_state
    sequence = np.zeros(period, dtype=int)

    for i in range(period):
        output_bit = state & 1
        sequence[i] = output_bit
        state >>= 1
        if output_bit:
            state ^= polynomial_hex

    return sequence


def to_bipolar(sequence):
    """Convert binary (0,1) to bipolar (-1,+1)"""
    return 2 * sequence - 1


def correlate_with_prbs(sense_signal, prbs_bipolar):
    """Correlate sense signal with PRBS code"""
    N = len(prbs_bipolar)
    sense_padded = np.tile(sense_signal, 2)
    correlation = np.zeros(N)

    for shift in range(N):
        window = sense_padded[shift:shift + N]
        correlation[shift] = np.dot(window, prbs_bipolar)

    return correlation


def gaussian(x, amplitude, center, width):
    """Gaussian function"""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))


print("="*80)
print("PROBLEM 2: COMPREHENSIVE VISUALIZATION PACKAGE")
print("="*80)

# Load data
notouch = np.loadtxt('HW3.Pr3.notouch.txt')
touch = np.loadtxt('HW3.Pr3.touch.txt')

# Generate PRBS
prbs_binary = generate_prbs(9, 0x110, initial_state=257)
prbs_bipolar = to_bipolar(prbs_binary)

# Correlations
correlation_notouch = correlate_with_prbs(notouch, prbs_bipolar)
correlation_touch = correlate_with_prbs(touch, prbs_bipolar)

# Find peaks
from scipy.signal import find_peaks
peaks, _ = find_peaks(np.abs(correlation_notouch), height=0, distance=50)
peak_heights = np.abs(correlation_notouch[peaks])
top5_indices = np.argsort(peak_heights)[-5:][::-1]
top5_peaks = np.sort(peaks[top5_indices])

# Get capacitances
cap_notouch = correlation_notouch[top5_peaks] / len(prbs_bipolar)
cap_touch = correlation_touch[top5_peaks] / len(prbs_bipolar)
cap_change = cap_touch - cap_notouch

physical_positions = np.array([5, 10, 15, 20, 25])
min_change_idx = np.argmin(cap_change)

# Gaussian fit
try:
    initial_guess = [cap_change[min_change_idx], physical_positions[min_change_idx], 5.0]
    popt, _ = curve_fit(gaussian, physical_positions, cap_change, p0=initial_guess, maxfev=5000)
    x_smooth = np.linspace(0, 30, 300)
    y_smooth = gaussian(x_smooth, *popt)
    gaussian_success = True
except:
    gaussian_success = False

# Noise estimation
mask = np.ones(len(correlation_notouch), dtype=bool)
for peak in top5_peaks:
    mask[max(0, peak-50):min(len(mask), peak+50)] = False
noise_samples = correlation_notouch[mask] / len(prbs_bipolar)
noise_std = np.std(noise_samples)

print(f"Creating comprehensive plots...")

# ============================================================================
# FIGURE 1: Individual Drive Line Analysis (5 subplots)
# ============================================================================

fig1 = plt.figure(figsize=(16, 10))

for i in range(5):
    ax = plt.subplot(2, 3, i+1)

    # Plot full correlation
    ax.plot(correlation_notouch, 'lightblue', linewidth=1, alpha=0.5, label='Other peaks')

    # Highlight this peak
    peak_pos = top5_peaks[i]
    window_start = max(0, peak_pos - 100)
    window_end = min(len(correlation_notouch), peak_pos + 100)

    ax.plot(range(window_start, window_end),
            correlation_notouch[window_start:window_end],
            'b-', linewidth=2, label='No touch')
    ax.plot(range(window_start, window_end),
            correlation_touch[window_start:window_end],
            'r-', linewidth=2, label='With touch')

    # Mark the peak
    ax.plot(peak_pos, correlation_notouch[peak_pos], 'bo', markersize=12,
            label=f'Peak: {correlation_notouch[peak_pos]:.1f}')
    ax.plot(peak_pos, correlation_touch[peak_pos], 'ro', markersize=12,
            label=f'Touch: {correlation_touch[peak_pos]:.1f}')

    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=peak_pos, color='g', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Delay (samples)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Correlation', fontsize=10, fontweight='bold')
    ax.set_title(f'Drive Line {i+1} (Position {physical_positions[i]} mm)\n' +
                 f'Phase offset: {peak_pos} samples, Cap change: {cap_change[i]:.4f}',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlim([window_start, window_end])

    # Add colored background based on change
    if i == min_change_idx:
        ax.set_facecolor('#ffcccc')  # Light red for touch location

# Overall title
fig1.suptitle('Problem 2: Individual Drive Line Correlation Analysis\n' +
              'Detailed view of each drive/sense pair',
              fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('problem2_individual_drives.png', dpi=150, bbox_inches='tight')
print("✓ Saved: problem2_individual_drives.png")

# ============================================================================
# FIGURE 2: Signal-to-Noise Analysis
# ============================================================================

fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Signal vs Noise levels
ax = axes[0, 0]
x_pos = np.arange(5)
signal_levels = np.abs(cap_change)
noise_level = np.full(5, noise_std)

bars1 = ax.bar(x_pos - 0.2, signal_levels, 0.4, label='Signal (cap change)',
               color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x_pos + 0.2, noise_level, 0.4, label='Noise level (σ)',
               color='coral', alpha=0.7, edgecolor='black', linewidth=1.5)

# Add SNR labels
for i, (sig, noise) in enumerate(zip(signal_levels, noise_level)):
    snr = sig / noise
    ax.text(i, max(sig, noise) * 1.1, f'SNR:\n{snr:.1f}',
            ha='center', fontsize=9, fontweight='bold')

ax.set_xlabel('Drive Line Position', fontsize=11, fontweight='bold')
ax.set_ylabel('Magnitude', fontsize=11, fontweight='bold')
ax.set_title('(a) Signal vs Noise Comparison', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{p} mm' for p in physical_positions])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: SNR bar chart
ax = axes[0, 1]
snr_values = signal_levels / noise_std
colors = ['red' if i == min_change_idx else 'steelblue' for i in range(5)]
bars = ax.bar(physical_positions, snr_values, width=3.5, color=colors,
              alpha=0.7, edgecolor='black', linewidth=2)

ax.axhline(y=10, color='orange', linestyle='--', linewidth=2,
           label='SNR = 10 (good threshold)')
ax.set_xlabel('Position (mm)', fontsize=11, fontweight='bold')
ax.set_ylabel('Signal-to-Noise Ratio', fontsize=11, fontweight='bold')
ax.set_title('(b) SNR at Each Position', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Annotate best SNR
best_snr_idx = np.argmax(snr_values)
ax.annotate(f'Best SNR: {snr_values[best_snr_idx]:.1f}',
            xy=(physical_positions[best_snr_idx], snr_values[best_snr_idx]),
            xytext=(physical_positions[best_snr_idx] + 5, snr_values[best_snr_idx] + 5),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'),
            fontsize=10, fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Plot 3: Processing Gain Demonstration
ax = axes[1, 0]

# Simulate what happens with different code lengths
code_lengths = [7, 31, 127, 511]
processing_gains = [np.sqrt(n) for n in code_lengths]

ax.plot(code_lengths, processing_gains, 'bo-', linewidth=3, markersize=10)
ax.axvline(x=511, color='r', linestyle='--', linewidth=2,
           label='PRBS511 (our code)')
ax.axhline(y=np.sqrt(511), color='r', linestyle='--', linewidth=2, alpha=0.5)

# Add annotation
ax.text(511, np.sqrt(511) + 1, f'PG = √511 = {np.sqrt(511):.1f}',
        fontsize=11, fontweight='bold', color='red',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax.set_xlabel('Code Length (bits)', fontsize=11, fontweight='bold')
ax.set_ylabel('Processing Gain (√N)', fontsize=11, fontweight='bold')
ax.set_title('(c) Processing Gain vs Code Length', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_xscale('log')

# Plot 4: Noise distribution with signal overlay
ax = axes[1, 1]

# Histogram of noise
ax.hist(noise_samples, bins=40, color='lightblue', edgecolor='black',
        alpha=0.7, density=True, label='Noise distribution')

# Overlay Gaussian fit
x_gauss = np.linspace(noise_samples.min(), noise_samples.max(), 100)
y_gauss = (1/(noise_std * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_gauss - np.mean(noise_samples))/noise_std)**2)
ax.plot(x_gauss, y_gauss, 'r-', linewidth=3, label='Gaussian fit')

# Mark signal locations
for i, cap in enumerate(cap_change):
    if i == min_change_idx:
        ax.axvline(x=cap, color='red', linestyle='-', linewidth=3,
                   label=f'Touch signal ({cap:.4f})', alpha=0.8)
    else:
        ax.axvline(x=cap, color='gray', linestyle='--', linewidth=1, alpha=0.5)

ax.set_xlabel('Value', fontsize=11, fontweight='bold')
ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
ax.set_title('(d) Noise Distribution with Signal Levels', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig2.suptitle('Problem 2: Signal-to-Noise Analysis and Processing Gain',
              fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('problem2_snr_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: problem2_snr_analysis.png")

# ============================================================================
# FIGURE 3: Capacitance Analysis
# ============================================================================

fig3, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Waterfall plot of capacitances
ax = axes[0, 0]
width = 0.25
x = np.arange(5)

bars1 = ax.bar(x - width, cap_notouch, width, label='No touch',
               color='steelblue', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x, cap_touch, width, label='With touch',
               color='coral', alpha=0.7, edgecolor='black')
bars3 = ax.bar(x + width, np.abs(cap_change), width, label='|Change|',
               color='green', alpha=0.7, edgecolor='black')

ax.set_xlabel('Drive Line', fontsize=11, fontweight='bold')
ax.set_ylabel('Capacitance', fontsize=11, fontweight='bold')
ax.set_title('(a) Capacitance Comparison', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'D{i+1}\n{p}mm' for i, p in enumerate(physical_positions)])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Percentage change
ax = axes[0, 1]
pct_change = 100 * cap_change / cap_notouch
colors = ['red' if i == min_change_idx else 'steelblue' for i in range(5)]

bars = ax.bar(physical_positions, pct_change, width=3.5, color=colors,
              alpha=0.7, edgecolor='black', linewidth=2)

ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax.set_xlabel('Position (mm)', fontsize=11, fontweight='bold')
ax.set_ylabel('Capacitance Change (%)', fontsize=11, fontweight='bold')
ax.set_title('(b) Percentage Capacitance Change', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (pos, pct) in enumerate(zip(physical_positions, pct_change)):
    ax.text(pos, pct - 2 if pct < 0 else pct + 2, f'{pct:.1f}%',
            ha='center', fontsize=9, fontweight='bold')

# Plot 3: Touch response spatial profile
ax = axes[1, 0]
ax.plot(physical_positions, cap_notouch, 'bo-', linewidth=2, markersize=10,
        label='Baseline (no touch)')
ax.plot(physical_positions, cap_touch, 'ro-', linewidth=2, markersize=10,
        label='With touch')

# Fill between to show change
ax.fill_between(physical_positions, cap_notouch, cap_touch,
                alpha=0.3, color='red', label='Capacitance reduction')

ax.set_xlabel('Position (mm)', fontsize=11, fontweight='bold')
ax.set_ylabel('Capacitance', fontsize=11, fontweight='bold')
ax.set_title('(c) Spatial Profile of Touch Response', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 4: Gaussian fit with confidence
ax = axes[1, 1]
ax.plot(physical_positions, cap_change, 'bo', markersize=12, label='Measured data')

if gaussian_success:
    ax.plot(x_smooth, y_smooth, 'r-', linewidth=3, label='Gaussian fit')

    # Add 1-sigma region
    center = popt[1]
    width = popt[2]
    ax.axvspan(center - width, center + width, alpha=0.2, color='red',
               label=f'±1σ region ({2*width:.1f} mm wide)')
    ax.axvline(x=center, color='green', linestyle='--', linewidth=2,
               label=f'Center: {center:.2f} mm')

ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax.set_xlabel('Position (mm)', fontsize=11, fontweight='bold')
ax.set_ylabel('Capacitance Change', fontsize=11, fontweight='bold')
ax.set_title('(d) Touch Localization with Gaussian Fit', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig3.suptitle('Problem 2: Detailed Capacitance Analysis',
              fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('problem2_capacitance_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: problem2_capacitance_analysis.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("COMPREHENSIVE PLOTS CREATED")
print("="*80)
print("\nThree detailed figure sets:")
print("  1. problem2_individual_drives.png")
print("     - Detailed view of each of the 5 drive lines")
print("     - Shows correlation peaks and touch effects individually")
print()
print("  2. problem2_snr_analysis.png")
print("     - Signal vs noise comparison")
print("     - SNR at each position")
print("     - Processing gain explanation")
print("     - Noise distribution with signal overlay")
print()
print("  3. problem2_capacitance_analysis.png")
print("     - Three-way capacitance comparison")
print("     - Percentage change analysis")
print("     - Spatial touch response profile")
print("     - Gaussian fit with confidence region")
print()
print("These plots provide complete visualization of all Problem 2 concepts!")
print("="*80)
