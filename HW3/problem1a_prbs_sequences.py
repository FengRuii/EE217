"""
Generate and save all PRBS sequences for Problem 1a
This creates a text file with all four PRBS sequences in readable format.
"""

import numpy as np


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


def format_sequence(sequence, name, bits_per_line=80):
    """Format a sequence nicely for printing"""
    output = []
    output.append("=" * 80)
    output.append(f"{name}")
    output.append(f"Length: {len(sequence)} bits")
    output.append(f"Period: 2^{int(np.log2(len(sequence) + 1))} - 1 = {len(sequence)}")
    output.append("=" * 80)
    output.append("")

    # Convert to string
    seq_str = ''.join(map(str, sequence))

    # Split into lines
    for i in range(0, len(seq_str), bits_per_line):
        chunk = seq_str[i:i + bits_per_line]
        output.append(chunk)

    output.append("")
    output.append(f"First 20 bits: {seq_str[:20]}")
    output.append(f"Last 20 bits:  {seq_str[-20:]}")
    output.append("")

    # Statistics
    num_ones = np.sum(sequence)
    num_zeros = len(sequence) - num_ones
    output.append(f"Statistics:")
    output.append(f"  Number of 1s: {num_ones}")
    output.append(f"  Number of 0s: {num_zeros}")
    output.append(f"  Ratio (1s/0s): {num_ones}/{num_zeros}")
    output.append("")

    return "\n".join(output)


# PRBS parameters from standard tables
prbs_configs = [
    (7, 0x60, "PRBS7"),
    (7, 0x44, "PRBS127 (using 7-bit LFSR with polynomial 0x44)"),
    (9, 0x110, "PRBS511 (using 9-bit LFSR)"),
    (10, 0x240, "PRBS1023 (using 10-bit LFSR)"),
]

# Wait, let me fix this - PRBS127 means 2^7-1 = 127 bits, which is just PRBS7
# Let me use the correct configs
prbs_configs = [
    (3, 0x3, "PRBS7 (2^3-1 = 7 bits)"),
    (7, 0x60, "PRBS127 (2^7-1 = 127 bits)"),
    (9, 0x110, "PRBS511 (2^9-1 = 511 bits)"),
    (10, 0x240, "PRBS1023 (2^10-1 = 1023 bits)"),
]

print("Generating PRBS sequences for Problem 1a...")
print()

# Generate all sequences
all_output = []
all_output.append("EE217 HOMEWORK 3 - PROBLEM 1A")
all_output.append("PRBS SEQUENCES")
all_output.append("=" * 80)
all_output.append("")

for n_bits, poly, name in prbs_configs:
    print(f"Generating {name}...")
    sequence = generate_prbs(n_bits, poly)
    formatted = format_sequence(sequence, name)
    all_output.append(formatted)
    all_output.append("\n" + "="*80 + "\n")

# Save to file
output_text = "\n".join(all_output)
filename = "problem1a_all_prbs_sequences.txt"

with open(filename, 'w') as f:
    f.write(output_text)

print(f"\n✓ All PRBS sequences saved to: {filename}")
print(f"  File size: {len(output_text)} characters")
print()
print("Summary:")
print("  - PRBS7:    7 bits")
print("  - PRBS127:  127 bits")
print("  - PRBS511:  511 bits")
print("  - PRBS1023: 1023 bits")
print(f"  Total: {7 + 127 + 511 + 1023} = 1668 bits")
