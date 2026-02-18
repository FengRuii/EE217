"""
Visual demonstration of orthogonality
Shows the difference between orthogonal and non-orthogonal sequences
"""

import numpy as np
import matplotlib.pyplot as plt

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# ============================================================================
# EXAMPLE 1: Perfectly Orthogonal Walsh Codes
# ============================================================================

# Walsh codes (4-bit)
walsh1 = np.array([+1, +1, +1, +1])
walsh2 = np.array([+1, -1, +1, -1])
walsh3 = np.array([+1, +1, -1, -1])

# Plot codes
ax = axes[0, 0]
x = np.arange(4)
ax.plot(x, walsh1, 'b-o', linewidth=2, markersize=8, label='Code 1')
ax.plot(x, walsh2, 'r-s', linewidth=2, markersize=8, label='Code 2')
ax.plot(x, walsh3, 'g-^', linewidth=2, markersize=8, label='Code 3')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlabel('Time', fontsize=11, fontweight='bold')
ax.set_ylabel('Value', fontsize=11, fontweight='bold')
ax.set_title('(a) Orthogonal Walsh Codes', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([-1.5, 1.5])

# Compute dot products
ax = axes[0, 1]
codes = [walsh1, walsh2, walsh3]
labels = ['Code 1', 'Code 2', 'Code 3']
n_codes = len(codes)

# Create dot product matrix
dot_matrix = np.zeros((n_codes, n_codes))
for i in range(n_codes):
    for j in range(n_codes):
        dot_matrix[i, j] = np.dot(codes[i], codes[j])

im = ax.imshow(dot_matrix, cmap='RdBu_r', vmin=-4, vmax=4)
ax.set_xticks(range(n_codes))
ax.set_yticks(range(n_codes))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_title('(b) Dot Product Matrix\n(Orthogonal = 0 off-diagonal)', fontsize=12, fontweight='bold')

# Add text annotations
for i in range(n_codes):
    for j in range(n_codes):
        text = ax.text(j, i, f'{dot_matrix[i, j]:.0f}',
                      ha="center", va="center", color="black", fontsize=14, fontweight='bold')

plt.colorbar(im, ax=ax)

# Show mixed signal and separation
ax = axes[0, 2]
mixed = walsh1 + 0.8*walsh2 + 0.5*walsh3  # Mix with different amplitudes
ax.plot(x, mixed, 'k-o', linewidth=2, markersize=8, label='Mixed signal')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlabel('Time', fontsize=11, fontweight='bold')
ax.set_ylabel('Value', fontsize=11, fontweight='bold')
ax.set_title('(c) Mixed Signal\n(all 3 codes transmitted together)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Add recovery information
recovery_text = 'Recover by correlation:\n'
recovery_text += f'mixed·code1 = {np.dot(mixed, walsh1):.1f} → amplitude = {np.dot(mixed, walsh1)/4:.2f}\n'
recovery_text += f'mixed·code2 = {np.dot(mixed, walsh2):.1f} → amplitude = {np.dot(mixed, walsh2)/4:.2f}\n'
recovery_text += f'mixed·code3 = {np.dot(mixed, walsh3):.1f} → amplitude = {np.dot(mixed, walsh3)/4:.2f}'
ax.text(0.02, 0.02, recovery_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ============================================================================
# EXAMPLE 2: Non-Orthogonal Sequences (Interference)
# ============================================================================

# Similar sequences (NOT orthogonal)
similar1 = np.array([+1, +1, +1, +1])
similar2 = np.array([+1, +1, +1, -1])  # Only 1 bit different
similar3 = np.array([+1, +1, -1, +1])  # Only 1 bit different

# Plot codes
ax = axes[1, 0]
ax.plot(x, similar1, 'b-o', linewidth=2, markersize=8, label='Code A')
ax.plot(x, similar2, 'r-s', linewidth=2, markersize=8, label='Code B')
ax.plot(x, similar3, 'g-^', linewidth=2, markersize=8, label='Code C')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlabel('Time', fontsize=11, fontweight='bold')
ax.set_ylabel('Value', fontsize=11, fontweight='bold')
ax.set_title('(d) Non-Orthogonal Codes', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([-1.5, 1.5])

# Compute dot products
ax = axes[1, 1]
codes_similar = [similar1, similar2, similar3]
labels_sim = ['Code A', 'Code B', 'Code C']

dot_matrix_sim = np.zeros((n_codes, n_codes))
for i in range(n_codes):
    for j in range(n_codes):
        dot_matrix_sim[i, j] = np.dot(codes_similar[i], codes_similar[j])

im = ax.imshow(dot_matrix_sim, cmap='RdBu_r', vmin=-4, vmax=4)
ax.set_xticks(range(n_codes))
ax.set_yticks(range(n_codes))
ax.set_xticklabels(labels_sim)
ax.set_yticklabels(labels_sim)
ax.set_title('(e) Dot Product Matrix\n(Large off-diagonal = INTERFERENCE)', fontsize=12, fontweight='bold')

# Add text annotations
for i in range(n_codes):
    for j in range(n_codes):
        text = ax.text(j, i, f'{dot_matrix_sim[i, j]:.0f}',
                      ha="center", va="center", color="black", fontsize=14, fontweight='bold')

plt.colorbar(im, ax=ax)

# Show mixed signal and failed separation
ax = axes[1, 2]
mixed_bad = similar1 + 0.8*similar2 + 0.5*similar3
ax.plot(x, mixed_bad, 'k-o', linewidth=2, markersize=8, label='Mixed signal')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlabel('Time', fontsize=11, fontweight='bold')
ax.set_ylabel('Value', fontsize=11, fontweight='bold')
ax.set_title('(f) Mixed Signal with Interference\n(codes interfere!)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Show interference problem
interference_text = 'Correlation gives WRONG results:\n'
interference_text += f'mixed·codeA = {np.dot(mixed_bad, similar1):.1f} → {np.dot(mixed_bad, similar1)/4:.2f} ✗\n'
interference_text += f'mixed·codeB = {np.dot(mixed_bad, similar2):.1f} → {np.dot(mixed_bad, similar2)/4:.2f} ✗\n'
interference_text += f'mixed·codeC = {np.dot(mixed_bad, similar3):.1f} → {np.dot(mixed_bad, similar3)/4:.2f} ✗\n'
interference_text += '\nExpected: 1.0, 0.8, 0.5\nInterference corrupts recovery!'
ax.text(0.02, 0.02, interference_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

# Overall title
fig.suptitle('Understanding Orthogonality: The Key to CDMA and Touch Sensors\n' +
             'Top Row: Orthogonal codes (perfect separation) | Bottom Row: Non-orthogonal codes (interference)',
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('orthogonality_explained.png', dpi=150, bbox_inches='tight')

print("✓ Saved: orthogonality_explained.png")
print("\nKey takeaways:")
print("  • Orthogonal codes: Dot product = 0 → Perfect separation")
print("  • Non-orthogonal codes: Large dot products → Interference")
print("  • Walsh codes: Perfectly orthogonal by design")
print("  • PRBS codes: Nearly orthogonal (good enough for practical use)")
