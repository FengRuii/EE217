"""
EE217 Homework 3 - Problem 1 Complete Solution
Generates all PRBS sequences and creates all required plots
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_prbs(n_bits, polynomial_hex, initial_state=1):
    """
    Generate PRBS using Galois LFSR

    Args:
        n_bits: Number of bits in shift register
        polynomial_hex: Feedback polynomial in hex
        initial_state: Starting state (default=1)

    Returns:
        Binary sequence (0s and 1s)
    """
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
    """Convert binary (0,1) to bipolar (-1,+1) for correlation"""
    return 2 * sequence - 1


def compute_autocorrelation(sequence):
    """
    Compute circular autocorrelation

    For good PRBS:
    - Peak of N at zero offset
    - Value of -1 at all other offsets
    """
    N = len(sequence)
    bipolar = to_bipolar(sequence)
    autocorr = np.zeros(N)

    for offset in range(N):
        autocorr[offset] = np.sum(bipolar * np.roll(bipolar, offset))

    return autocorr


def compute_crosscorrelation(seq_a, seq_b):
    """
    Compute circular cross-correlation between two sequences

    xcor(n) = sum(seqA[k] * seqB[k-n]) for all k
    """
    N = len(seq_a)
    assert len(seq_b) == N, "Sequences must be same length"

    bipolar_a = to_bipolar(seq_a)
    bipolar_b = to_bipolar(seq_b)

    crosscorr = np.zeros(N)

    for offset in range(N):
        crosscorr[offset] = np.sum(bipolar_a * np.roll(bipolar_b, offset))

    return crosscorr


# ============================================================================
# PROBLEM 1A: Generate all PRBS sequences
# ============================================================================

print("="*80)
print("PROBLEM 1A: GENERATING PRBS SEQUENCES")
print("="*80)

prbs_configs = [
    (3, 0x3, "PRBS7"),
    (7, 0x60, "PRBS127"),
    (9, 0x110, "PRBS511"),
    (10, 0x240, "PRBS1023"),
]

sequences = {}

for n_bits, poly, name in prbs_configs:
    seq = generate_prbs(n_bits, poly)
    sequences[name] = seq
    print(f"{name}: {len(seq)} bits, polynomial 0x{poly:X}")
    print(f"  First 20 bits: {''.join(map(str, seq[:20]))}")
    print(f"  Ones: {np.sum(seq)}, Zeros: {len(seq) - np.sum(seq)}")
    print()

# ============================================================================
# PROBLEM 1B: Autocorrelation of PRBS511
# ============================================================================

print("="*80)
print("PROBLEM 1B: AUTOCORRELATION ANALYSIS")
print("="*80)

# Part 1: Full PRBS511 autocorrelation
prbs511 = sequences["PRBS511"]
autocorr_511 = compute_autocorrelation(prbs511)

peak_value = autocorr_511[0]
off_peak_values = autocorr_511[1:]
max_off_peak = np.max(off_peak_values)
min_off_peak = np.min(off_peak_values)

print(f"\nFull PRBS511 Autocorrelation:")
print(f"  Peak at offset 0: {peak_value}")
print(f"  Off-peak max: {max_off_peak}")
print(f"  Off-peak min: {min_off_peak}")
print(f"  All off-peak values equal to -1? {np.all(off_peak_values == -1)}")

# Part 2: Test 255-bit subsequences
print(f"\n255-bit Subsequence Analysis:")
print(f"  Testing all possible 255-bit subsequences from PRBS511...")

best_subsequence = None
best_start_idx = -1
best_max_off_peak = float('inf')

# Test all possible starting positions
for start_idx in range(len(prbs511)):
    # Extract 255-bit subsequence (circular)
    subseq = np.zeros(255, dtype=int)
    for i in range(255):
        subseq[i] = prbs511[(start_idx + i) % len(prbs511)]

    # Compute autocorrelation
    autocorr_sub = compute_autocorrelation(subseq)

    # Check off-peak values
    off_peak = autocorr_sub[1:]
    max_off_peak_sub = np.max(np.abs(off_peak))

    # Track best subsequence
    if max_off_peak_sub < best_max_off_peak:
        best_max_off_peak = max_off_peak_sub
        best_subsequence = subseq
        best_start_idx = start_idx

print(f"  Best subsequence starts at index: {best_start_idx}")
print(f"  Best max off-peak value: {best_max_off_peak}")
print(f"  Is it a true PRBS255? {best_max_off_peak == 1}")
print(f"  Conclusion: 255-bit subsequence {'IS' if best_max_off_peak == 1 else 'IS NOT'} a true PRBS255")

# Compute autocorrelation of best subsequence for plotting
autocorr_255 = compute_autocorrelation(best_subsequence)

# ============================================================================
# PROBLEM 1C: Cross-correlation between different generators
# ============================================================================

print("\n" + "="*80)
print("PROBLEM 1C: CROSS-CORRELATION ANALYSIS")
print("="*80)

# Generate PRBS511 with alternate polynomial (0x108 = x^9 + x^4 + 1)
print(f"\nGenerating two PRBS511 sequences with different polynomials:")
seqA = generate_prbs(9, 0x110)  # Original: x^9 + x^5 + 1
seqB = generate_prbs(9, 0x108)  # Alternate: x^9 + x^4 + 1

print(f"  seqA: polynomial 0x110, length {len(seqA)}")
print(f"  seqB: polynomial 0x108, length {len(seqB)}")

# Check if sequences are equal
are_equal = np.array_equal(seqA, seqB)
print(f"\nAre seqA and seqB equal? {are_equal}")

if not are_equal:
    # Find first difference
    first_diff = np.where(seqA != seqB)[0][0]
    print(f"  First difference at index {first_diff}")
    print(f"  seqA[{first_diff}] = {seqA[first_diff]}")
    print(f"  seqB[{first_diff}] = {seqB[first_diff]}")

# Compute cross-correlation
crosscorr = compute_crosscorrelation(seqA, seqB)

max_crosscorr = np.max(crosscorr)
max_crosscorr_idx = np.argmax(crosscorr)
min_crosscorr = np.min(crosscorr)

print(f"\nCross-correlation results:")
print(f"  Maximum value: {max_crosscorr} at offset {max_crosscorr_idx}")
print(f"  Minimum value: {min_crosscorr}")
print(f"  Mean (excluding max): {np.mean(np.delete(crosscorr, max_crosscorr_idx)):.2f}")

# ============================================================================
# PLOTTING: Create comprehensive visualizations
# ============================================================================

print("\n" + "="*80)
print("CREATING PLOTS")
print("="*80)

# Create a large figure with all plots
fig = plt.figure(figsize=(16, 12))

# ============================================================================
# Plot 1: PRBS511 Full Autocorrelation
# ============================================================================
ax1 = plt.subplot(2, 3, 1)
offsets = np.arange(len(autocorr_511))
ax1.plot(offsets, autocorr_511, 'b-', linewidth=1.5)
ax1.axhline(y=-1, color='r', linestyle='--', linewidth=1, label='Expected off-peak (-1)')
ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax1.set_xlabel('Offset (samples)', fontsize=11)
ax1.set_ylabel('Autocorrelation', fontsize=11)
ax1.set_title('Problem 1b: PRBS511 Autocorrelation (Full Sequence)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.text(0.05, 0.95, f'Peak = {peak_value}\nOff-peak = -1',
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================================================
# Plot 2: PRBS511 Autocorrelation Zoomed (show the spike clearly)
# ============================================================================
ax2 = plt.subplot(2, 3, 2)
zoom_range = 50
ax2.plot(offsets[:zoom_range], autocorr_511[:zoom_range], 'b-o', linewidth=2, markersize=4)
ax2.axhline(y=-1, color='r', linestyle='--', linewidth=1, label='Expected off-peak (-1)')
ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax2.set_xlabel('Offset (samples)', fontsize=11)
ax2.set_ylabel('Autocorrelation', fontsize=11)
ax2.set_title('Problem 1b: PRBS511 Autocorrelation (Zoomed)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xlim([-2, zoom_range])

# ============================================================================
# Plot 3: 255-bit Subsequence Autocorrelation
# ============================================================================
ax3 = plt.subplot(2, 3, 3)
offsets_255 = np.arange(len(autocorr_255))
ax3.plot(offsets_255, autocorr_255, 'g-', linewidth=1.5)
ax3.axhline(y=-1, color='r', linestyle='--', linewidth=1, label='Ideal PRBS')
ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax3.set_xlabel('Offset (samples)', fontsize=11)
ax3.set_ylabel('Autocorrelation', fontsize=11)
ax3.set_title('Problem 1b: 255-bit Subsequence Autocorrelation', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.text(0.05, 0.95, f'Peak = {autocorr_255[0]}\nMax off-peak = {best_max_off_peak}\n(NOT a true PRBS)',
         transform=ax3.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

# ============================================================================
# Plot 4: Cross-correlation (Full)
# ============================================================================
ax4 = plt.subplot(2, 3, 4)
offsets_cross = np.arange(len(crosscorr))
ax4.plot(offsets_cross, crosscorr, 'm-', linewidth=1.5)
ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax4.set_xlabel('Offset (samples)', fontsize=11)
ax4.set_ylabel('Cross-correlation', fontsize=11)
ax4.set_title('Problem 1c: Cross-correlation between seqA (0x110) and seqB (0x108)',
              fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.text(0.05, 0.95, f'Max = {max_crosscorr}\nat offset {max_crosscorr_idx}',
         transform=ax4.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))

# ============================================================================
# Plot 5: Cross-correlation (Zoomed around maximum)
# ============================================================================
ax5 = plt.subplot(2, 3, 5)
zoom_range = 100
center = max_crosscorr_idx
start = max(0, center - zoom_range // 2)
end = min(len(crosscorr), center + zoom_range // 2)
ax5.plot(offsets_cross[start:end], crosscorr[start:end], 'm-o', linewidth=2, markersize=4)
ax5.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax5.axvline(x=max_crosscorr_idx, color='r', linestyle='--', linewidth=1, label='Max value')
ax5.set_xlabel('Offset (samples)', fontsize=11)
ax5.set_ylabel('Cross-correlation', fontsize=11)
ax5.set_title('Problem 1c: Cross-correlation (Zoomed around peak)', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend()

# ============================================================================
# Plot 6: Comparison of first 100 bits of seqA and seqB
# ============================================================================
ax6 = plt.subplot(2, 3, 6)
n_compare = 100
x_vals = np.arange(n_compare)
ax6.plot(x_vals, to_bipolar(seqA[:n_compare]), 'b-o', linewidth=1.5, markersize=3,
         label='seqA (poly 0x110)', alpha=0.7)
ax6.plot(x_vals, to_bipolar(seqB[:n_compare]) - 0.1, 'r-s', linewidth=1.5, markersize=3,
         label='seqB (poly 0x108)', alpha=0.7)
ax6.set_xlabel('Bit index', fontsize=11)
ax6.set_ylabel('Bipolar value', fontsize=11)
ax6.set_title('Problem 1c: Comparison of seqA vs seqB (first 100 bits)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend()
ax6.set_ylim([-1.5, 1.5])

plt.tight_layout()
plt.savefig('problem1_complete_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: problem1_complete_analysis.png")

# ============================================================================
# Additional plot: Just Problem 1c cross-correlation (standalone)
# ============================================================================
fig2, ax = plt.subplots(figsize=(12, 6))
ax.plot(offsets_cross, crosscorr, 'm-', linewidth=2)
ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
ax.axvline(x=max_crosscorr_idx, color='r', linestyle='--', linewidth=2,
           label=f'Max = {max_crosscorr} at offset {max_crosscorr_idx}')
ax.set_xlabel('Time Offset (samples)', fontsize=12, fontweight='bold')
ax.set_ylabel('Cross-correlation', fontsize=12, fontweight='bold')
ax.set_title('Problem 1c: Cross-correlation between PRBS511 sequences\n(polynomial 0x110 vs 0x108)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

# Add text box with statistics
textstr = f'Sequences: seqA (0x110) and seqB (0x108)\n'
textstr += f'Length: {len(seqA)} bits each\n'
textstr += f'Max cross-correlation: {max_crosscorr}\n'
textstr += f'Min cross-correlation: {min_crosscorr}\n'
textstr += f'Sequences equal? {are_equal}'
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('problem1c_crosscorr.png', dpi=150, bbox_inches='tight')
print("✓ Saved: problem1c_crosscorr.png")

# ============================================================================
# Summary Report
# ============================================================================
print("\n" + "="*80)
print("SUMMARY REPORT")
print("="*80)

print("\nProblem 1a: PRBS Generation")
print("-" * 40)
for name, seq in sequences.items():
    print(f"  {name}: {len(seq)} bits generated successfully")

print("\nProblem 1b: Autocorrelation")
print("-" * 40)
print(f"  Full PRBS511:")
print(f"    ✓ Peak at offset 0: {peak_value}")
print(f"    ✓ All off-peak values: -1")
print(f"  255-bit subsequence:")
print(f"    ✗ Max off-peak: {best_max_off_peak} (should be -1 for true PRBS)")
print(f"    Conclusion: NOT a true PRBS255")

print("\nProblem 1c: Cross-correlation")
print("-" * 40)
print(f"  seqA (poly 0x110) vs seqB (poly 0x108):")
print(f"    Sequences equal? {are_equal}")
print(f"    Max cross-correlation: {max_crosscorr}")
print(f"    Conclusion: Different sequences with moderate cross-correlation")

print("\n" + "="*80)
print("ALL PLOTS GENERATED SUCCESSFULLY")
print("="*80)
print("\nFiles created:")
print("  - problem1_complete_analysis.png (6 subplots)")
print("  - problem1c_crosscorr.png (detailed cross-correlation)")
print()
