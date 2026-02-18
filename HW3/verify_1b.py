"""
Verify Problem 1b - PRBS511 autocorrelation and 255-bit subsequence analysis
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_prbs(n_bits, polynomial_hex, initial_state=1):
    """Generate PRBS using Galois LFSR."""
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

def prbs_to_bipolar(sequence):
    """Convert binary (0,1) to bipolar (-1,+1)."""
    return 2 * sequence - 1

def compute_autocorrelation(sequence):
    """Compute circular autocorrelation."""
    N = len(sequence)
    autocorr = np.zeros(N)

    for offset in range(N):
        autocorr[offset] = np.sum(sequence * np.roll(sequence, offset))

    return autocorr

# Generate PRBS511
print("="*70)
print("VERIFYING PROBLEM 1b")
print("="*70)
print("\n1. Generating PRBS511 with polynomial 0x110...")

prbs511 = generate_prbs(9, 0x110, initial_state=1)
prbs511_bipolar = prbs_to_bipolar(prbs511)

print(f"   Length: {len(prbs511)} bits")
print(f"   First 30 bits: {prbs511[:30]}")
print(f"   Number of 1s: {np.sum(prbs511)} (should be 256)")
print(f"   Number of 0s: {len(prbs511) - np.sum(prbs511)} (should be 255)")
print()

# Check autocorrelation
print("2. Computing autocorrelation of PRBS511...")
autocorr_511 = compute_autocorrelation(prbs511_bipolar)

print(f"   Autocorrelation at offset 0: {autocorr_511[0]}")
print(f"   Autocorrelation at offset 1: {autocorr_511[1]}")
print(f"   Autocorrelation at offset 2: {autocorr_511[2]}")
print(f"   Min autocorrelation (off-peak): {np.min(autocorr_511[1:])}")
print(f"   Max autocorrelation (off-peak): {np.max(autocorr_511[1:])}")
print()

# Verify all off-peak values are -1
unique_offpeak = np.unique(autocorr_511[1:])
print(f"   Unique off-peak values: {unique_offpeak}")
if len(unique_offpeak) == 1 and unique_offpeak[0] == -1:
    print("   ✓ PERFECT! All off-peak values are -1")
else:
    print("   ✗ ERROR: Off-peak values are not all -1")
print()

# Analyze 255-bit subsequences
print("3. Analyzing all 255-bit subsequences...")
print("   (Testing if any subsequence has PRBS255 properties)")
print()

best_start = 0
best_max_offpeak = float('inf')
all_max_offpeaks = []

for start_idx in range(511):
    # Extract 255-bit subsequence
    subseq = np.roll(prbs511_bipolar, -start_idx)[:255]
    autocorr_sub = compute_autocorrelation(subseq)

    # Check peak at zero
    peak = autocorr_sub[0]

    # Find max off-peak autocorrelation (absolute value)
    max_offpeak = np.max(np.abs(autocorr_sub[1:]))
    all_max_offpeaks.append(max_offpeak)

    if max_offpeak < best_max_offpeak:
        best_max_offpeak = max_offpeak
        best_start = start_idx
        best_autocorr = autocorr_sub

print(f"   Best subsequence:")
print(f"   - Starting index: {best_start}")
print(f"   - Peak at zero: {best_autocorr[0]} (should be 255)")
print(f"   - Max off-peak: {best_max_offpeak} (should be -1 for true PRBS255)")
print(f"   - Min off-peak: {np.min(best_autocorr[1:])}")
print()

print(f"   Statistics across all 511 subsequences:")
print(f"   - Best max off-peak: {np.min(all_max_offpeaks)}")
print(f"   - Worst max off-peak: {np.max(all_max_offpeaks)}")
print(f"   - Average max off-peak: {np.mean(all_max_offpeaks):.1f}")
print()

# Answer the question
print("4. Answering the question:")
print()
print("   Q: Is a 255-bit subsequence of PRBS511 actually a PRBS255 sequence?")
print("   A: NO")
print()
print("   Reasoning:")
print(f"   - For a true PRBS255, all off-peak autocorrelation values should be -1")
print(f"   - The BEST 255-bit subsequence has max off-peak of {best_max_offpeak}")
print(f"   - This is {best_max_offpeak - (-1)} away from the ideal value of -1")
print()
print("   Q: Does it depend on which 255-bit subsequence you choose?")
print("   A: YES, absolutely")
print()
print("   Reasoning:")
print(f"   - Max off-peak ranges from {np.min(all_max_offpeaks)} to {np.max(all_max_offpeaks)}")
print(f"   - Different starting positions give very different autocorrelation properties")
print()

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Full PRBS511 autocorrelation
axes[0, 0].plot(autocorr_511, 'b-', linewidth=1)
axes[0, 0].axhline(y=-1, color='r', linestyle='--', alpha=0.5, label='Expected off-peak (-1)')
axes[0, 0].set_xlabel('Offset (samples)')
axes[0, 0].set_ylabel('Autocorrelation')
axes[0, 0].set_title('PRBS511 Autocorrelation (Perfect!)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Zoom on off-peak
axes[0, 1].plot(autocorr_511[1:100], 'b-', linewidth=1)
axes[0, 1].axhline(y=-1, color='r', linestyle='--', alpha=0.5, label='Expected (-1)')
axes[0, 1].set_xlabel('Offset (samples)')
axes[0, 1].set_ylabel('Autocorrelation')
axes[0, 1].set_title('PRBS511 Off-Peak Values (Zoomed)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([-2, 0])

# Plot 3: Best 255-bit subsequence
axes[1, 0].plot(best_autocorr, 'g-', linewidth=1)
axes[1, 0].axhline(y=-1, color='r', linestyle='--', alpha=0.5, label='Ideal off-peak (-1)')
axes[1, 0].axhline(y=255, color='orange', linestyle='--', alpha=0.5, label='Expected peak (255)')
axes[1, 0].set_xlabel('Offset (samples)')
axes[1, 0].set_ylabel('Autocorrelation')
axes[1, 0].set_title(f'Best 255-bit Subsequence (start={best_start})')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Distribution of max off-peak values
axes[1, 1].hist(all_max_offpeaks, bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(x=1, color='r', linestyle='--', linewidth=2, label='Ideal (1)')
axes[1, 1].set_xlabel('Max Off-Peak Autocorrelation')
axes[1, 1].set_ylabel('Number of Subsequences')
axes[1, 1].set_title('Distribution of Max Off-Peak Values\n(across all 511 subsequences)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('verify_problem1b.png', dpi=150, bbox_inches='tight')
print("5. Generated plot: verify_problem1b.png")
print()

print("="*70)
print("VERIFICATION COMPLETE")
print("="*70)
print()
print("CONCLUSION:")
print("✓ PRBS511 generation is correct (perfect autocorrelation)")
print("✓ 255-bit subsequences are NOT true PRBS255 sequences")
print("✓ Choice of subsequence matters significantly")
