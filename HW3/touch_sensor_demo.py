"""
Complete Touch Sensor Simulation
Shows exactly how PRBS/CDMA separates capacitive sensor signals

Run this to see the concepts in action!
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("CAPACITIVE TOUCH SENSOR WITH CDMA - COMPLETE SIMULATION")
print("="*70)

# ============================================================================
# PART 1: SETUP
# ============================================================================

print("\n" + "="*70)
print("PART 1: SYSTEM SETUP")
print("="*70)

N = 4  # Code length (4-bit Walsh codes for simplicity)
num_drives = 3

# Walsh codes (perfectly orthogonal)
codes = np.array([
    [+1, +1, +1, +1],  # Drive 1 - Position 1cm
    [+1, -1, +1, -1],  # Drive 2 - Position 2cm
    [+1, +1, -1, -1],  # Drive 3 - Position 3cm
])

print(f"\nNumber of drive lines: {num_drives}")
print(f"Code length: {N}")
print(f"\nDrive codes:")
for i in range(num_drives):
    code_str = ''.join(['+' if x > 0 else '-' for x in codes[i]])
    print(f"  Drive {i+1} (at {i+1}cm): {code_str}")

# Verify orthogonality
print(f"\nVerifying orthogonality:")
for i in range(num_drives):
    for j in range(i+1, num_drives):
        dot = np.dot(codes[i], codes[j])
        print(f"  Drive{i+1} · Drive{j+1} = {dot} ✓")

# ============================================================================
# PART 2: NO TOUCH BASELINE
# ============================================================================

print("\n" + "="*70)
print("PART 2: BASELINE MEASUREMENT (NO TOUCH)")
print("="*70)

# Capacitances without touch (in pF)
cap_notouch = np.array([10, 8, 12])

print(f"\nBaseline capacitances:")
for i in range(num_drives):
    print(f"  Drive {i+1}: {cap_notouch[i]} pF")

# Generate transmitted signals
print(f"\nTransmitted signals over {N} time samples:")
print(f"{'Time':<8} {'Drive1':<8} {'Drive2':<8} {'Drive3':<8}")
print("-"*40)
for t in range(N):
    print(f"t={t:<6} {codes[0,t]:+2.0f}       {codes[1,t]:+2.0f}       {codes[2,t]:+2.0f}")

# Generate sense signal (matrix multiplication)
sense_notouch = codes.T @ cap_notouch

print(f"\nSense line receives mixed signal:")
print(f"  Sense = C1×Drive1 + C2×Drive2 + C3×Drive3")
print(f"\nMeasured values:")
for t in range(N):
    calculation = f"{cap_notouch[0]}×{codes[0,t]:+.0f} + {cap_notouch[1]}×{codes[1,t]:+.0f} + {cap_notouch[2]}×{codes[2,t]:+.0f}"
    print(f"  t={t}: {calculation} = {sense_notouch[t]:+.0f}")

print(f"\nComplete sense signal: {sense_notouch}")

# ============================================================================
# PART 3: CORRELATION - EXTRACTING CAPACITANCES
# ============================================================================

print("\n" + "="*70)
print("PART 3: EXTRACTING CAPACITANCES BY CORRELATION")
print("="*70)

recovered_notouch = []

for i in range(num_drives):
    print(f"\n--- Recovering Drive {i+1} capacitance ---")
    print(f"Correlate sense with Drive{i+1}'s code:")

    print(f"\n  Sense:       {sense_notouch}")
    code_str = ''.join([f'{x:+2.0f} ' for x in codes[i]])
    print(f"  Drive{i+1} code: [{code_str[:-1]}]")

    print(f"\n  Multiply element-by-element:")
    products = sense_notouch * codes[i]
    for t in range(N):
        print(f"    {sense_notouch[t]:+.0f} × {codes[i,t]:+.0f} = {products[t]:+.0f}")

    correlation = np.sum(products)
    recovered = correlation / N
    recovered_notouch.append(recovered)

    print(f"\n  Sum: {correlation:.0f}")
    print(f"  Divide by N={N}: {correlation:.0f}/{N} = {recovered:.1f} pF")
    print(f"  True value: {cap_notouch[i]} pF")
    print(f"  ✓ Perfect match!")

print(f"\n" + "="*70)
print(f"BASELINE SUMMARY:")
for i in range(num_drives):
    print(f"  Position {i+1}cm: {recovered_notouch[i]:.1f} pF")
print("="*70)

# ============================================================================
# PART 4: WITH TOUCH
# ============================================================================

print("\n" + "="*70)
print("PART 4: FINGER TOUCHES AT 2cm POSITION")
print("="*70)

# Capacitances with touch at position 2cm
cap_touch = np.array([10, 3, 10])

print(f"\nCapacitance changes due to finger:")
print(f"{'Position':<12} {'Before':<12} {'After':<12} {'Change'}")
print("-"*50)
for i in range(num_drives):
    change = cap_touch[i] - cap_notouch[i]
    marker = " ← TOUCH!" if i == 1 else ""
    print(f"{i+1}cm          {cap_notouch[i]} pF        {cap_touch[i]} pF        {change:+.0f} pF{marker}")

# Generate new sense signal
sense_touch = codes.T @ cap_touch

print(f"\nNew sense signal with touch:")
print(f"  Before: {sense_notouch}")
print(f"  After:  {sense_touch}")
print(f"  The signal changed!")

# Recover capacitances
print(f"\nRecovering new capacitances:")

recovered_touch = []
for i in range(num_drives):
    correlation = np.dot(sense_touch, codes[i])
    recovered = correlation / N
    recovered_touch.append(recovered)

    change = recovered - recovered_notouch[i]
    marker = " ← BIGGEST CHANGE" if abs(change) == max(abs(np.array(recovered_touch) - np.array(recovered_notouch[:len(recovered_touch)]))) and len(recovered_touch) == num_drives else ""

    print(f"  Drive {i+1}: {recovered:.1f} pF (was {recovered_notouch[i]:.1f} pF, change: {change:+.1f} pF){marker}")

# ============================================================================
# PART 5: TOUCH DETECTION
# ============================================================================

print("\n" + "="*70)
print("PART 5: TOUCH LOCATION DETECTION")
print("="*70)

differences = np.array(recovered_touch) - np.array(recovered_notouch)

print(f"\nCapacitance changes:")
print(f"{'Position':<12} {'Change':<12} {'Interpretation'}")
print("-"*50)
for i in range(num_drives):
    if abs(differences[i]) < 1:
        interpretation = "No touch"
    elif abs(differences[i]) > 4:
        interpretation = "DIRECT TOUCH"
    else:
        interpretation = "Partial effect"

    print(f"{i+1}cm          {differences[i]:+.1f} pF     {interpretation}")

touch_position = np.argmin(differences) + 1
print(f"\n{'='*70}")
print(f"DETECTED: Finger touching at position {touch_position}cm!")
print(f"{'='*70}")

# ============================================================================
# PART 6: WITH NOISE
# ============================================================================

print("\n" + "="*70)
print("PART 6: ADDING REALISTIC NOISE")
print("="*70)

np.random.seed(42)  # Reproducible results
noise_std = 2.0
noise = np.random.randn(N) * noise_std

sense_noisy = sense_touch + noise

print(f"\nAdding Gaussian noise (σ = {noise_std}):")
print(f"  Clean signal: {sense_touch}")
print(f"  Noise:        {noise}")
print(f"  Noisy signal: {sense_noisy}")

print(f"\nRecovering capacitances from noisy signal:")
recovered_noisy = []
for i in range(num_drives):
    correlation = np.dot(sense_noisy, codes[i])
    recovered = correlation / N
    recovered_noisy.append(recovered)

    error = recovered - cap_touch[i]
    print(f"  Drive {i+1}: {recovered:.2f} pF (true: {cap_touch[i]} pF, error: {error:+.2f} pF)")

print(f"\nNoise effect:")
print(f"  Maximum error: {max(abs(np.array(recovered_noisy) - cap_touch)):.2f} pF")
print(f"  Still clearly shows touch at position 2cm!")

# Processing gain
print(f"\n  Processing gain = √{N} = {np.sqrt(N):.2f}")
print(f"  With longer codes (N=511): √511 = 22.6")
print(f"  → 27 dB noise suppression!")

# ============================================================================
# PART 7: VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("PART 7: GENERATING VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Plot 1: Drive codes
axes[0, 0].set_title('Drive Line Codes (Walsh Codes)', fontsize=12, fontweight='bold')
for i in range(num_drives):
    axes[0, 0].plot(codes[i], 'o-', label=f'Drive {i+1} ({i+1}cm)', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Time sample')
axes[0, 0].set_ylabel('Code value')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([-1.5, 1.5])

# Plot 2: Sense signals comparison
axes[0, 1].set_title('Sense Line Signals', fontsize=12, fontweight='bold')
axes[0, 1].plot(sense_notouch, 'b-o', label='No touch', linewidth=2, markersize=8)
axes[0, 1].plot(sense_touch, 'r-s', label='With touch', linewidth=2, markersize=8)
axes[0, 1].plot(sense_noisy, 'g-^', label='With noise', linewidth=2, markersize=6, alpha=0.7)
axes[0, 1].set_xlabel('Time sample')
axes[0, 1].set_ylabel('Sense signal')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Baseline capacitances
positions = [1, 2, 3]
axes[1, 0].set_title('Baseline Capacitances', fontsize=12, fontweight='bold')
axes[1, 0].bar(positions, cap_notouch, color='steelblue', alpha=0.7, edgecolor='black', linewidth=2)
axes[1, 0].set_xlabel('Position (cm)')
axes[1, 0].set_ylabel('Capacitance (pF)')
axes[1, 0].set_xticks(positions)
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(cap_notouch):
    axes[1, 0].text(positions[i], v + 0.5, f'{v} pF', ha='center', fontweight='bold')

# Plot 4: With touch
axes[1, 1].set_title('Capacitances With Touch at 2cm', fontsize=12, fontweight='bold')
width = 0.35
x = np.array(positions)
axes[1, 1].bar(x - width/2, cap_notouch, width, label='No touch', color='steelblue', alpha=0.7, edgecolor='black')
axes[1, 1].bar(x + width/2, cap_touch, width, label='With touch', color='coral', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Position (cm)')
axes[1, 1].set_ylabel('Capacitance (pF)')
axes[1, 1].set_xticks(positions)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Plot 5: Capacitance changes
axes[2, 0].set_title('Capacitance Change (Touch Detection)', fontsize=12, fontweight='bold')
colors = ['green' if abs(d) < 1 else 'red' if abs(d) > 4 else 'orange' for d in differences]
bars = axes[2, 0].bar(positions, differences, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[2, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[2, 0].set_xlabel('Position (cm)')
axes[2, 0].set_ylabel('Capacitance change (pF)')
axes[2, 0].set_xticks(positions)
axes[2, 0].grid(True, alpha=0.3, axis='y')
for i, (p, v) in enumerate(zip(positions, differences)):
    axes[2, 0].text(p, v - 0.5 if v < 0 else v + 0.5, f'{v:.1f} pF',
                    ha='center', fontweight='bold', fontsize=10)
# Add annotation for touch
axes[2, 0].annotate('TOUCH\nDETECTED!', xy=(2, differences[1]), xytext=(2.5, -3),
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                   fontsize=12, fontweight='bold', color='red')

# Plot 6: Correlation demonstration
axes[2, 1].set_title('How Correlation Separates Signals', fontsize=12, fontweight='bold')
x_pos = np.arange(num_drives)
correlations_before = recovered_notouch
correlations_after = recovered_noisy
axes[2, 1].bar(x_pos - 0.2, correlations_before, 0.4, label='Clean (no touch)',
               color='steelblue', alpha=0.7, edgecolor='black')
axes[2, 1].bar(x_pos + 0.2, correlations_after, 0.4, label='Noisy (with touch)',
               color='coral', alpha=0.7, edgecolor='black')
axes[2, 1].set_xlabel('Drive Line')
axes[2, 1].set_ylabel('Recovered Capacitance (pF)')
axes[2, 1].set_xticks(x_pos)
axes[2, 1].set_xticklabels([f'Drive{i+1}\n({i+1}cm)' for i in range(num_drives)])
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('touch_sensor_complete_demo.png', dpi=150, bbox_inches='tight')
print("\nSaved: touch_sensor_complete_demo.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY: THE COMPLETE PROCESS")
print("="*70)

print("""
1. SETUP:
   - 3 drive lines with orthogonal Walsh codes
   - Each drive at different position (1cm, 2cm, 3cm)

2. TRANSMIT:
   - All drives transmit simultaneously
   - Signals mix on sense line

3. MEASURE:
   - ADC samples mixed sense signal
   - Contains contributions from all drives

4. CORRELATE:
   - Multiply sense by each drive's code
   - Sum products and divide by N
   - Extracts each individual capacitance

5. DETECT TOUCH:
   - Compare to baseline
   - Biggest change shows touch location
   - Works even with noise!

6. PROCESSING GAIN:
   - Correlation averages out noise
   - Gain = √N
   - Longer codes = better noise resistance

KEY INSIGHT:
Orthogonal codes let us transmit everything at once,
then separate cleanly with correlation!

This is exactly how your smartphone touch screen works,
just with more drive lines and longer codes (PRBS511).
""")

print("="*70)
print("SIMULATION COMPLETE!")
print("="*70)
print("\nCheck out touch_sensor_complete_demo.png to see all the plots!")
