"""
EE217 Homework 3 - Problem 2 Complete Solution
CDMA-Based Touch Sensors
Creates comprehensive visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import signal


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
    """
    Correlate sense signal with PRBS code
    Returns correlation at all circular shifts
    """
    N = len(prbs_bipolar)

    # Pad sense signal to make it work with all shifts
    sense_padded = np.tile(sense_signal, 2)  # Repeat signal twice for circular correlation

    correlation = np.zeros(N)

    for shift in range(N):
        # Extract a window of length N starting at shift
        window = sense_padded[shift:shift + N]
        # Compute dot product
        correlation[shift] = np.dot(window, prbs_bipolar)

    return correlation


def gaussian(x, amplitude, center, width):
    """Gaussian function for fitting"""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))


print("="*80)
print("PROBLEM 2: CDMA-BASED TOUCH SENSORS")
print("="*80)

# ============================================================================
# Load data files
# ============================================================================

print("\nLoading data files...")
notouch = np.loadtxt('HW3.Pr3.notouch.txt')
touch = np.loadtxt('HW3.Pr3.touch.txt')

print(f"  No-touch data: {len(notouch)} samples")
print(f"  Touch data: {len(touch)} samples")

# ============================================================================
# Generate PRBS511 sequence
# ============================================================================

print("\nGenerating PRBS511 reference code...")
prbs_binary = generate_prbs(9, 0x110, initial_state=257)
prbs_bipolar = to_bipolar(prbs_binary)

print(f"  PRBS length: {len(prbs_bipolar)} bits")
print(f"  Polynomial: 0x110")
print(f"  Initial state: 257")

# ============================================================================
# PART A: Find correlation peaks (no touch baseline)
# ============================================================================

print("\n" + "="*80)
print("PART A: BASELINE CORRELATION (NO TOUCH)")
print("="*80)

# Correlate no-touch signal with PRBS
correlation_notouch = correlate_with_prbs(notouch, prbs_bipolar)

# Find peaks
from scipy.signal import find_peaks
peaks, properties = find_peaks(np.abs(correlation_notouch), height=0, distance=50)

# Sort by peak height and get top 5
peak_heights = np.abs(correlation_notouch[peaks])
top5_indices = np.argsort(peak_heights)[-5:][::-1]  # Get top 5, sorted descending
top5_peaks = peaks[top5_indices]
top5_peaks_sorted = np.sort(top5_peaks)  # Sort by position

print(f"\nFound 5 correlation peaks:")
print(f"{'Peak #':<8} {'Position':<12} {'Correlation':<15} {'Capacitance':<15}")
print("-"*60)

capacitances_notouch = []
positions_notouch = []

for i, peak_pos in enumerate(top5_peaks_sorted):
    corr_value = correlation_notouch[peak_pos]
    # Normalize by sequence length to get capacitance
    capacitance = corr_value / len(prbs_bipolar)
    capacitances_notouch.append(capacitance)
    positions_notouch.append(peak_pos)
    print(f"{i+1:<8} {peak_pos:<12} {corr_value:<15.2f} {capacitance:<15.4f}")

# ============================================================================
# PART B: Touch detection
# ============================================================================

print("\n" + "="*80)
print("PART B: TOUCH DETECTION")
print("="*80)

# Correlate touch signal
correlation_touch = correlate_with_prbs(touch, prbs_bipolar)

print(f"\nFinding correlation peaks in touch signal...")
capacitances_touch = []

for i, peak_pos in enumerate(top5_peaks_sorted):
    corr_value = correlation_touch[peak_pos]
    capacitance = corr_value / len(prbs_bipolar)
    capacitances_touch.append(capacitance)

# Compute changes
capacitances_notouch = np.array(capacitances_notouch)
capacitances_touch = np.array(capacitances_touch)
capacitance_changes = capacitances_touch - capacitances_notouch

print(f"\n{'Position':<10} {'No Touch':<12} {'Touch':<12} {'Change':<12}")
print("-"*50)

for i in range(5):
    print(f"{i+1:<10} {capacitances_notouch[i]:<12.4f} {capacitances_touch[i]:<12.4f} {capacitance_changes[i]:<12.4f}")

# Find touch location
min_change_idx = np.argmin(capacitance_changes)
print(f"\n*** Biggest capacitance change at position {min_change_idx + 1} ***")
print(f"    Change: {capacitance_changes[min_change_idx]:.4f}")

# Map to physical coordinates
# Position with smallest delay → 5mm, second → 10mm, etc.
physical_positions = np.array([5, 10, 15, 20, 25])  # in mm

# Sort by peak position (delay) to map to physical location
position_to_physical = dict(zip(range(5), physical_positions))

touch_position_mm = physical_positions[min_change_idx]
print(f"    Physical location: {touch_position_mm} mm")

# Fit Gaussian to find precise location
try:
    # Use physical positions and capacitance changes for fitting
    initial_guess = [capacitance_changes[min_change_idx], touch_position_mm, 5.0]
    popt, pcov = curve_fit(gaussian, physical_positions, capacitance_changes,
                          p0=initial_guess, maxfev=5000)

    amplitude_fit, center_fit, width_fit = popt
    print(f"\nGaussian fit:")
    print(f"    Center: {center_fit:.2f} mm")
    print(f"    Width (σ): {width_fit:.2f} mm")
    print(f"    Amplitude: {amplitude_fit:.4f}")

    # Generate smooth curve for plotting
    x_smooth = np.linspace(0, 30, 300)
    y_smooth = gaussian(x_smooth, *popt)

    gaussian_fit_success = True
except Exception as e:
    print(f"\nGaussian fit failed: {e}")
    print("Using peak location as touch position")
    center_fit = touch_position_mm
    gaussian_fit_success = False

# ============================================================================
# Noise estimation
# ============================================================================

print("\n" + "="*80)
print("NOISE ESTIMATION")
print("="*80)

# Noise is estimated from correlation values away from peaks
# Create mask to exclude peak regions
mask = np.ones(len(correlation_notouch), dtype=bool)
for peak in top5_peaks_sorted:
    # Exclude ±50 samples around each peak
    mask[max(0, peak-50):min(len(mask), peak+50)] = False

noise_samples = correlation_notouch[mask] / len(prbs_bipolar)
noise_std = np.std(noise_samples)
noise_mean = np.mean(noise_samples)

print(f"Noise statistics (away from peaks):")
print(f"  Mean: {noise_mean:.6f}")
print(f"  Std dev: {noise_std:.6f}")
print(f"  Noise per sample: {noise_std:.6f}")

# ============================================================================
# CREATE COMPREHENSIVE PLOTS
# ============================================================================

print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

fig = plt.figure(figsize=(16, 12))

# ============================================================================
# Plot 1: Raw signals
# ============================================================================
ax1 = plt.subplot(3, 3, 1)
time_axis = np.arange(len(notouch))
ax1.plot(time_axis[:2000], notouch[:2000], 'b-', linewidth=0.5, alpha=0.7, label='No touch')
ax1.set_xlabel('Sample', fontsize=10)
ax1.set_ylabel('Amplitude', fontsize=10)
ax1.set_title('(a) Raw Sense Signal - No Touch', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2 = plt.subplot(3, 3, 2)
ax2.plot(time_axis[:2000], touch[:2000], 'r-', linewidth=0.5, alpha=0.7, label='With touch')
ax2.set_xlabel('Sample', fontsize=10)
ax2.set_ylabel('Amplitude', fontsize=10)
ax2.set_title('(b) Raw Sense Signal - With Touch', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# ============================================================================
# Plot 3: PRBS code
# ============================================================================
ax3 = plt.subplot(3, 3, 3)
ax3.plot(prbs_bipolar[:100], 'g-o', linewidth=1.5, markersize=3)
ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax3.set_xlabel('Bit index', fontsize=10)
ax3.set_ylabel('Value', fontsize=10)
ax3.set_title('(c) PRBS511 Reference Code (first 100 bits)', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_ylim([-1.5, 1.5])

# ============================================================================
# Plot 4: Correlation - No Touch (full)
# ============================================================================
ax4 = plt.subplot(3, 3, 4)
ax4.plot(correlation_notouch, 'b-', linewidth=1, alpha=0.7)
ax4.plot(top5_peaks_sorted, correlation_notouch[top5_peaks_sorted], 'ro',
         markersize=10, label='5 peaks (drive lines)')
ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax4.set_xlabel('Delay (samples)', fontsize=10)
ax4.set_ylabel('Correlation', fontsize=10)
ax4.set_title('(d) Correlation - No Touch (5 peaks found)', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()

# ============================================================================
# Plot 5: Correlation - Touch (full)
# ============================================================================
ax5 = plt.subplot(3, 3, 5)
ax5.plot(correlation_touch, 'r-', linewidth=1, alpha=0.7)
ax5.plot(top5_peaks_sorted, correlation_touch[top5_peaks_sorted], 'go',
         markersize=10, label='5 peaks')
ax5.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax5.set_xlabel('Delay (samples)', fontsize=10)
ax5.set_ylabel('Correlation', fontsize=10)
ax5.set_title('(e) Correlation - With Touch', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend()

# ============================================================================
# Plot 6: Comparison of peaks
# ============================================================================
ax6 = plt.subplot(3, 3, 6)
x_pos = np.arange(5)
width = 0.35
ax6.bar(x_pos - width/2, capacitances_notouch, width, label='No touch',
        color='steelblue', alpha=0.7, edgecolor='black')
ax6.bar(x_pos + width/2, capacitances_touch, width, label='With touch',
        color='coral', alpha=0.7, edgecolor='black')
ax6.set_xlabel('Drive Line', fontsize=10)
ax6.set_ylabel('Capacitance (normalized)', fontsize=10)
ax6.set_title('(f) Capacitance Comparison', fontsize=11, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels([f'D{i+1}' for i in range(5)])
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Plot 7: Capacitance changes
# ============================================================================
ax7 = plt.subplot(3, 3, 7)
colors = ['red' if i == min_change_idx else 'steelblue' for i in range(5)]
bars = ax7.bar(physical_positions, capacitance_changes, width=3.5,
               color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax7.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax7.set_xlabel('Physical Position (mm)', fontsize=10)
ax7.set_ylabel('Capacitance Change', fontsize=10)
ax7.set_title('(g) Capacitance Change vs Position', fontsize=11, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

# Annotate the touch location
ax7.annotate(f'TOUCH!\n{touch_position_mm} mm',
             xy=(touch_position_mm, capacitance_changes[min_change_idx]),
             xytext=(touch_position_mm + 5, capacitance_changes[min_change_idx] - 0.05),
             arrowprops=dict(arrowstyle='->', lw=2, color='red'),
             fontsize=10, fontweight='bold', color='red',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# ============================================================================
# Plot 8: Gaussian fit
# ============================================================================
ax8 = plt.subplot(3, 3, 8)
ax8.plot(physical_positions, capacitance_changes, 'bo-', markersize=10,
         linewidth=2, label='Measured data')
if gaussian_fit_success:
    ax8.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Gaussian fit')
    ax8.axvline(x=center_fit, color='g', linestyle='--', linewidth=2,
                label=f'Center: {center_fit:.2f} mm')
ax8.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax8.set_xlabel('Position (mm)', fontsize=10)
ax8.set_ylabel('Capacitance Change', fontsize=10)
ax8.set_title('(h) Gaussian Fit for Touch Location', fontsize=11, fontweight='bold')
ax8.grid(True, alpha=0.3)
ax8.legend(fontsize=9)

# ============================================================================
# Plot 9: Noise distribution
# ============================================================================
ax9 = plt.subplot(3, 3, 9)
ax9.hist(noise_samples, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax9.axvline(x=noise_mean, color='r', linestyle='--', linewidth=2,
            label=f'Mean: {noise_mean:.4f}')
ax9.axvline(x=noise_mean + noise_std, color='orange', linestyle='--', linewidth=1,
            label=f'±σ: {noise_std:.4f}')
ax9.axvline(x=noise_mean - noise_std, color='orange', linestyle='--', linewidth=1)
ax9.set_xlabel('Normalized Correlation Value', fontsize=10)
ax9.set_ylabel('Frequency', fontsize=10)
ax9.set_title('(i) Noise Distribution', fontsize=11, fontweight='bold')
ax9.legend(fontsize=9)
ax9.grid(True, alpha=0.3, axis='y')

# Overall title
fig.suptitle('Problem 2: CDMA-Based Touch Sensor - Complete Analysis',
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('problem2_complete_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: problem2_complete_analysis.png")

# ============================================================================
# Create a focused touch detection plot
# ============================================================================

fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Capacitance changes with Gaussian fit
ax = axes[0]
ax.plot(physical_positions, capacitance_changes, 'bo-', markersize=12,
        linewidth=3, label='Measured', zorder=3)
if gaussian_fit_success:
    ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Gaussian fit', alpha=0.7)
    ax.axvline(x=center_fit, color='g', linestyle='--', linewidth=2,
               label=f'Touch at {center_fit:.2f} mm')
ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax.set_xlabel('Position (mm)', fontsize=12, fontweight='bold')
ax.set_ylabel('Capacitance Change', fontsize=12, fontweight='bold')
ax.set_title('Touch Location Detection', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# Shade the touch region
if gaussian_fit_success:
    touch_region = np.abs(x_smooth - center_fit) < width_fit
    ax.fill_between(x_smooth[touch_region], y_smooth[touch_region],
                     alpha=0.2, color='red', label='Touch region (±σ)')

# Right: Bar chart comparison
ax = axes[1]
x_pos = np.arange(5)
ax.bar(x_pos, capacitances_notouch, 0.4, label='Baseline (no touch)',
       color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
ax.bar(x_pos, capacitances_touch, 0.4, label='With touch',
       color='coral', alpha=0.5, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Drive Line Position', fontsize=12, fontweight='bold')
ax.set_ylabel('Capacitance', fontsize=12, fontweight='bold')
ax.set_title('Capacitance at Each Drive Line', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{pos} mm' for pos in physical_positions])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Highlight the touched position
ax.text(min_change_idx, capacitances_touch[min_change_idx] + 0.01,
        '← Touch here!', fontsize=10, fontweight='bold', color='red',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('problem2_touch_detection.png', dpi=150, bbox_inches='tight')
print("✓ Saved: problem2_touch_detection.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\nPart A: Baseline Correlation Peaks")
print("-"*50)
for i, (pos, cap) in enumerate(zip(positions_notouch, capacitances_notouch)):
    print(f"  Drive {i+1}: Delay = {pos} samples, Capacitance = {cap:.4f}")

print("\nPart B: Touch Detection")
print("-"*50)
print(f"  Touch location: {touch_position_mm} mm (Drive {min_change_idx + 1})")
if gaussian_fit_success:
    print(f"  Gaussian fit center: {center_fit:.2f} mm")
    print(f"  Gaussian width: {width_fit:.2f} mm")
print(f"  Maximum capacitance change: {capacitance_changes[min_change_idx]:.4f}")

print("\nNoise Analysis")
print("-"*50)
print(f"  Noise standard deviation: {noise_std:.6f}")
print(f"  Signal-to-noise ratio: {np.abs(capacitance_changes[min_change_idx]) / noise_std:.1f}")

print("\n" + "="*80)
print("ALL PLOTS GENERATED SUCCESSFULLY!")
print("="*80)
print("\nFiles created:")
print("  - problem2_complete_analysis.png (9 subplots)")
print("  - problem2_touch_detection.png (focused 2-plot)")
