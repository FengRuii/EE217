"""
Problem 1c: Three-plot visualization
Cross-correlation analysis between two PRBS511 generators
"""

import numpy as np
import matplotlib.pyplot as plt


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


def compute_crosscorrelation(seq_a, seq_b):
    """Compute circular cross-correlation"""
    N = len(seq_a)
    bipolar_a = to_bipolar(seq_a)
    bipolar_b = to_bipolar(seq_b)
    crosscorr = np.zeros(N)

    for offset in range(N):
        crosscorr[offset] = np.sum(bipolar_a * np.roll(bipolar_b, offset))

    return crosscorr


# Generate sequences
print("Generating PRBS511 sequences...")
seqA = generate_prbs(9, 0x110)  # x^9 + x^5 + 1
seqB = generate_prbs(9, 0x108)  # x^9 + x^4 + 1

print(f"seqA: polynomial 0x110, length {len(seqA)}")
print(f"seqB: polynomial 0x108, length {len(seqB)}")

# Compute cross-correlation
crosscorr = compute_crosscorrelation(seqA, seqB)

max_val = np.max(crosscorr)
max_idx = np.argmax(crosscorr)
min_val = np.min(crosscorr)

print(f"\nCross-correlation analysis:")
print(f"  Maximum: {max_val} at offset {max_idx}")
print(f"  Minimum: {min_val}")

# Create figure with 3 subplots
fig = plt.figure(figsize=(16, 5))

# ============================================================================
# Plot 1: Full Cross-Correlation
# ============================================================================
ax1 = plt.subplot(1, 3, 1)
offsets = np.arange(len(crosscorr))
ax1.plot(offsets, crosscorr, 'b-', linewidth=1.5, alpha=0.8)
ax1.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.3)
ax1.axhline(y=max_val, color='r', linestyle='--', linewidth=1, alpha=0.5, label=f'Max = {max_val}')
ax1.axhline(y=min_val, color='r', linestyle='--', linewidth=1, alpha=0.5, label=f'Min = {min_val}')
ax1.scatter([max_idx], [max_val], color='red', s=100, zorder=5, label=f'Peak at offset {max_idx}')

ax1.set_xlabel('Time Offset (samples)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Cross-correlation', fontsize=12, fontweight='bold')
ax1.set_title('(a) Full Cross-Correlation', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# Add statistics box
stats_text = f'Length: {len(seqA)} bits\nMax: {max_val}\nMin: {min_val}\nMean: {np.mean(crosscorr):.2f}'
ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ============================================================================
# Plot 2: Histogram of Cross-Correlation Values
# ============================================================================
ax2 = plt.subplot(1, 3, 2)
ax2.hist(crosscorr, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(x=max_val, color='r', linestyle='--', linewidth=2, label=f'Max = {max_val}')
ax2.axvline(x=min_val, color='r', linestyle='--', linewidth=2, label=f'Min = {min_val}')
ax2.axvline(x=0, color='k', linestyle='-', linewidth=1, alpha=0.5)

ax2.set_xlabel('Cross-correlation Value', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency (count)', fontsize=12, fontweight='bold')
ax2.set_title('(b) Distribution of Cross-Correlation Values', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend(fontsize=9)

# Add statistics
std_dev = np.std(crosscorr)
dist_text = f'σ = {std_dev:.2f}\nRange: [{min_val:.0f}, {max_val:.0f}]'
ax2.text(0.98, 0.97, dist_text, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# ============================================================================
# Plot 3: Direct Sequence Comparison (First 150 bits)
# ============================================================================
ax3 = plt.subplot(1, 3, 3)
n_compare = 150
x_vals = np.arange(n_compare)

# Convert to bipolar for visualization
bipolar_a = to_bipolar(seqA[:n_compare])
bipolar_b = to_bipolar(seqB[:n_compare])

# Plot as stem plot
markerline_a, stemlines_a, baseline_a = ax3.stem(x_vals, bipolar_a, linefmt='b-',
                                                   markerfmt='bo', basefmt='k-',
                                                   label='seqA (0x110)')
markerline_a.set_markersize(3)
stemlines_a.set_linewidth(1)
stemlines_a.set_alpha(0.6)

# Offset seqB slightly for visibility
offset_y = -0.15
markerline_b, stemlines_b, baseline_b = ax3.stem(x_vals, bipolar_b + offset_y,
                                                   linefmt='r-', markerfmt='rs',
                                                   basefmt='none',
                                                   label='seqB (0x108, offset)')
markerline_b.set_markersize(3)
stemlines_b.set_linewidth(1)
stemlines_b.set_alpha(0.6)

ax3.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.3)
ax3.set_xlabel('Bit Index', fontsize=12, fontweight='bold')
ax3.set_ylabel('Bipolar Value', fontsize=12, fontweight='bold')
ax3.set_title(f'(c) Sequence Comparison (First {n_compare} bits)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9, loc='upper right')
ax3.set_ylim([-1.5, 1.5])

# Highlight differences
differences = np.where(seqA[:n_compare] != seqB[:n_compare])[0]
num_diffs = len(differences)
diff_text = f'Differences: {num_diffs}/{n_compare}\n({100*num_diffs/n_compare:.1f}%)'
ax3.text(0.02, 0.97, diff_text, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.8))

# Overall title
fig.suptitle('Problem 1c: Cross-Correlation Analysis of Two PRBS511 Generators\n' +
             'seqA (polynomial 0x110) vs seqB (polynomial 0x108)',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('problem1c_analysis.png', dpi=150, bbox_inches='tight')

print("\n✓ Saved: problem1c_analysis.png")
print("\nPlots created:")
print("  (a) Full cross-correlation showing all 511 offsets")
print("  (b) Histogram showing distribution of cross-correlation values")
print("  (c) Direct bit-by-bit comparison of the two sequences")
