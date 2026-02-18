"""
Interactive demonstrations of key concepts from HW3
Run each section to see the concepts in action!
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("INTERACTIVE DEMOS - EE217 HW3 CONCEPTS")
print("="*70)
print()

# ============================================================================
# DEMO 1: How LFSR generates PRBS
# ============================================================================

print("\n" + "="*70)
print("DEMO 1: Step-by-step LFSR operation")
print("="*70)

def show_lfsr_steps(n_bits=5, polynomial=0x12, initial_state=1, num_steps=10):
    """Show detailed LFSR operation step by step."""

    print(f"\nGenerating PRBS with {n_bits}-bit LFSR")
    print(f"Polynomial: {hex(polynomial)} = {bin(polynomial)}")
    print(f"Initial state: {initial_state} = {bin(initial_state)}")
    print()

    state = initial_state
    sequence = []

    print(f"{'Step':<6} {'State (binary)':<15} {'State (dec)':<12} {'Output':<8} {'Feedback'}")
    print("-"*70)

    for step in range(num_steps):
        # Show current state
        state_binary = format(state, f'0{n_bits}b')
        output_bit = state & 1

        # Calculate feedback for Galois LFSR
        feedback = output_bit

        # Show the step
        print(f"{step:<6} {state_binary:<15} {state:<12} {output_bit:<8} {'XOR' if feedback else 'none'}")

        sequence.append(output_bit)

        # Update state
        state >>= 1
        if feedback:
            state ^= polynomial

    print()
    print(f"Generated sequence: {sequence}")
    print(f"Sequence as bits: {''.join(map(str, sequence))}")
    print()

show_lfsr_steps(n_bits=5, polynomial=0x12, initial_state=1, num_steps=15)

# Show full period
print("Full period example (PRBS7):")
from hw3_solution import generate_prbs
prbs7 = generate_prbs(7, 0x60, initial_state=1)
print(f"Length: {len(prbs7)}")
print(f"First 31 bits: {''.join(map(str, prbs7[:31].astype(int)))}")
print(f"Next 31 bits:  {''.join(map(str, prbs7[31:62].astype(int)))}")
print()


# ============================================================================
# DEMO 2: Autocorrelation visualization
# ============================================================================

print("\n" + "="*70)
print("DEMO 2: Understanding autocorrelation")
print("="*70)

def visualize_correlation_process():
    """Show how correlation works step by step."""

    # Simple 7-bit sequence
    seq = np.array([1, 0, 1, 1, 0, 0, 1])
    seq_bipolar = 2*seq - 1  # Convert to +1/-1

    print("\nOriginal sequence: ", seq)
    print("Bipolar format:    ", seq_bipolar)
    print()

    # Show correlation at different shifts
    N = len(seq)
    print(f"{'Shift':<8} {'Aligned sequences':<50} {'Products':<30} {'Sum'}")
    print("-"*100)

    for shift in range(min(N, 7)):
        shifted = np.roll(seq_bipolar, shift)
        products = seq_bipolar * shifted
        correlation = np.sum(products)

        seq_str = ''.join(['+' if x==1 else '-' for x in seq_bipolar])
        shift_str = ''.join(['+' if x==1 else '-' for x in shifted])
        prod_str = ''.join([f'{p:+2d}' for p in products])

        print(f"{shift:<8} {seq_str:<25} {shift_str:<25} {prod_str:<30} {correlation:+3d}")

    print()
    print("Notice:")
    print("- Shift 0: All positions match → maximum correlation")
    print("- Other shifts: Some match, some don't → lower correlation")
    print()

visualize_correlation_process()

# Plot autocorrelation for PRBS31
prbs31 = generate_prbs(5, 0x12, initial_state=1)
prbs31_bipolar = 2*prbs31 - 1

autocorr = np.zeros(31)
for shift in range(31):
    autocorr[shift] = np.sum(prbs31_bipolar * np.roll(prbs31_bipolar, shift))

plt.figure(figsize=(12, 4))
plt.stem(autocorr, basefmt=' ')
plt.axhline(y=-1, color='r', linestyle='--', alpha=0.5, label='Expected off-peak (-1)')
plt.xlabel('Shift')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of PRBS31 (5-bit LFSR)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('demo_autocorrelation.png', dpi=150)
print("Saved: demo_autocorrelation.png")
print()


# ============================================================================
# DEMO 3: CDMA visualization
# ============================================================================

print("\n" + "="*70)
print("DEMO 3: CDMA - Multiple signals at once")
print("="*70)

def demo_cdma():
    """Show how CDMA separates multiple signals."""

    # Three users with different PRBS phases
    prbs = generate_prbs(5, 0x12, initial_state=1)
    prbs_bipolar = 2*prbs - 1

    user1_code = prbs_bipolar
    user2_code = np.roll(prbs_bipolar, 10)
    user3_code = np.roll(prbs_bipolar, 20)

    # Each user sends a different "bit" (amplitude)
    user1_data = 5.0   # User 1 sends amplitude 5
    user2_data = 3.0   # User 2 sends amplitude 3
    user3_data = -2.0  # User 3 sends amplitude -2

    # Combined signal (what the receiver sees)
    combined = user1_data*user1_code + user2_data*user2_code + user3_data*user3_code

    # Add noise
    noise = np.random.randn(31) * 0.5
    received = combined + noise

    print("Transmitted signals:")
    print(f"User 1: amplitude = {user1_data}")
    print(f"User 2: amplitude = {user2_data}")
    print(f"User 3: amplitude = {user3_data}")
    print()

    # Receiver correlates with each user's code
    corr1 = np.sum(received * user1_code) / 31
    corr2 = np.sum(received * user2_code) / 31
    corr3 = np.sum(received * user3_code) / 31

    print("Recovered signals (using correlation):")
    print(f"User 1: {corr1:.2f} (true: {user1_data})")
    print(f"User 2: {corr2:.2f} (true: {user2_data})")
    print(f"User 3: {corr3:.2f} (true: {user3_data})")
    print()
    print("Even though all signals were mixed together,")
    print("correlation successfully separates them!")
    print()

    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    axes[0].plot(user1_code * user1_data, 'b-', label='User 1', linewidth=1.5, alpha=0.7)
    axes[0].plot(user2_code * user2_data, 'r-', label='User 2', linewidth=1.5, alpha=0.7)
    axes[0].plot(user3_code * user3_data, 'g-', label='User 3', linewidth=1.5, alpha=0.7)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Individual User Signals')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(combined, 'purple', linewidth=1.5)
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Combined Signal (all mixed together)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(received, 'k-', linewidth=1, alpha=0.7, label='Received')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_xlabel('Sample')
    axes[2].set_title('Received Signal (combined + noise)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_cdma.png', dpi=150)
    print("Saved: demo_cdma.png")
    print()

demo_cdma()


# ============================================================================
# DEMO 4: Time-domain aliasing
# ============================================================================

print("\n" + "="*70)
print("DEMO 4: Time-domain aliasing (frequency sampling)")
print("="*70)

def demo_time_aliasing():
    """Visualize time-domain aliasing from frequency sampling."""

    a = 0.8

    # True signal
    n_true = np.arange(-30, 31)
    x_true = a**np.abs(n_true)

    # DTFT
    omega = np.linspace(0, 2*np.pi, 1000)
    X_omega = (1 - a**2) / (1 - 2*a*np.cos(omega) + a**2)

    # Sample at N points and IFFT
    N_values = [10, 20, 50, 100]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, N in enumerate(N_values):
        # Sample frequency domain
        k = np.arange(N)
        omega_k = 2 * np.pi * k / N
        X_k = (1 - a**2) / (1 - 2*a*np.cos(omega_k) + a**2)

        # IFFT
        x_reconstructed = np.fft.ifft(X_k).real

        # Plot
        axes[idx].plot(n_true, x_true, 'b-', alpha=0.3, linewidth=3, label='True signal')
        axes[idx].stem(np.arange(N), x_reconstructed, 'r', basefmt=' ', label='IFFT result')
        axes[idx].set_xlabel('n')
        axes[idx].set_ylabel('x(n)')
        axes[idx].set_title(f'N = {N} frequency samples')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(-5, max(N+5, 35))

        # Calculate error at n=0
        error = abs(x_reconstructed[0] - 1.0)
        axes[idx].text(0.02, 0.98, f'Error at n=0: {error:.4f}',
                      transform=axes[idx].transAxes,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('demo_time_aliasing.png', dpi=150)
    print("Saved: demo_time_aliasing.png")
    print()

    print("Observation:")
    print("- N=10: Signal hasn't decayed, severe aliasing")
    print("- N=20: Still significant aliasing (your homework case)")
    print("- N=50: Getting better")
    print("- N=100: Good reconstruction near n=0")
    print()
    print("Key insight: Need enough frequency samples so the time signal")
    print("             fits within one period!")
    print()

demo_time_aliasing()


# ============================================================================
# DEMO 5: Processing gain from correlation
# ============================================================================

print("\n" + "="*70)
print("DEMO 5: Processing gain - correlation fights noise")
print("="*70)

def demo_processing_gain():
    """Show how correlation provides processing gain against noise."""

    # PRBS31
    prbs31 = generate_prbs(5, 0x12, initial_state=1)
    prbs31_bipolar = 2*prbs31 - 1

    # Signal: weak PRBS buried in noise
    signal_amplitude = 0.5
    signal = signal_amplitude * prbs31_bipolar

    # Noise: much stronger than signal!
    np.random.seed(42)
    noise = np.random.randn(31) * 2.0

    # Received signal
    received = signal + noise

    print(f"Signal amplitude: {signal_amplitude}")
    print(f"Noise std dev: 2.0")
    print(f"SNR before correlation: {signal_amplitude / 2.0:.2f} ({20*np.log10(signal_amplitude/2.0):.1f} dB)")
    print()

    # Correlate to detect
    correlation = np.sum(received * prbs31_bipolar)
    detected_amplitude = correlation / 31

    print(f"Detected amplitude: {detected_amplitude:.3f}")
    print(f"Error: {abs(detected_amplitude - signal_amplitude):.3f}")
    print()

    # Theoretical processing gain
    processing_gain = np.sqrt(31)
    print(f"Theoretical processing gain: √31 = {processing_gain:.2f}")
    print(f"SNR improvement: {20*np.log10(processing_gain):.1f} dB")
    print()

    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    axes[0].plot(signal, 'b-', linewidth=2, label='True signal')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Original Signal (weak)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(received, 'r-', linewidth=1, alpha=0.7, label='Signal + Noise')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Received Signal (buried in noise!)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Show correlation at all shifts
    corr_all = np.zeros(31)
    for shift in range(31):
        corr_all[shift] = np.sum(received * np.roll(prbs31_bipolar, shift)) / 31

    axes[2].stem(corr_all, basefmt=' ')
    axes[2].axhline(y=signal_amplitude, color='g', linestyle='--',
                    linewidth=2, label=f'True amplitude ({signal_amplitude})')
    axes[2].set_ylabel('Correlation')
    axes[2].set_xlabel('Shift')
    axes[2].set_title('Correlation Output (signal recovered!)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_processing_gain.png', dpi=150)
    print("Saved: demo_processing_gain.png")
    print()
    print("Notice: Even though the signal was completely buried in noise,")
    print("        correlation successfully recovered it!")
    print()

demo_processing_gain()


# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("DEMOS COMPLETE!")
print("="*70)
print()
print("Generated visualizations:")
print("  - demo_autocorrelation.png")
print("  - demo_cdma.png")
print("  - demo_time_aliasing.png")
print("  - demo_processing_gain.png")
print()
print("These demos showed:")
print("  1. How LFSRs generate PRBS step-by-step")
print("  2. How autocorrelation works")
print("  3. How CDMA separates multiple signals")
print("  4. Why time-domain aliasing occurs")
print("  5. How correlation provides processing gain")
print()
print("Read CONCEPTS_EXPLAINED.md for detailed theory!")
print()
