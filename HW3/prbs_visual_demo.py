"""
Visual PRBS Generator - See It Step by Step!
Shows exactly how PRBS works with simple animations
"""

import time
import sys

def print_slowly(text, delay=0.03):
    """Print text with a typewriter effect"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def visualize_lfsr(n_bits=3, polynomial=0x3, initial_state=0b101, steps=10):
    """
    Visualize LFSR operation step by step with animations
    """
    print("="*70)
    print("PRBS GENERATOR - VISUAL DEMONSTRATION")
    print("="*70)
    print()

    print_slowly(f"⚙️  Setting up a {n_bits}-bit LFSR...")
    print(f"   Initial state: {bin(initial_state)} = {initial_state}")
    print(f"   Polynomial: {hex(polynomial)} = {bin(polynomial)}")
    print()

    input("Press Enter to start generating...")
    print()

    state = initial_state
    sequence = []

    # Visual representation
    box_width = 7

    for step in range(steps):
        print("="*70)
        print(f"STEP {step + 1}")
        print("="*70)
        print()

        # Show current state
        print("Current state:")
        bits = [(state >> i) & 1 for i in range(n_bits-1, -1, -1)]

        # Draw boxes
        print("  " + "┌" + "─"*box_width + "┐" + " " + "┌" + "─"*box_width + "┐" + " " + "┌" + "─"*box_width + "┐")
        print("  " + "│" + f"  {bits[0]}    ".center(box_width) + "│" + " " + "│" + f"  {bits[1]}    ".center(box_width) + "│" + " " + "│" + f"  {bits[2]}    ".center(box_width) + "│")
        print("  " + "└" + "─"*box_width + "┘" + " " + "└" + "─"*box_width + "┘" + " " + "└" + "─"*box_width + "┘")
        print("    Bit 2         Bit 1         Bit 0")
        print()

        # Output bit
        output_bit = state & 1
        print(f"📤 Output bit (rightmost): {output_bit}")
        sequence.append(output_bit)
        print(f"   Sequence so far: {sequence}")
        print()

        time.sleep(0.5)

        # Shift
        print("➡️  Shifting all bits right...")
        state_after_shift = state >> 1
        bits_after = [(state_after_shift >> i) & 1 for i in range(n_bits-1, -1, -1)]

        print()
        print("  After shift:")
        print("  " + "┌" + "─"*box_width + "┐" + " " + "┌" + "─"*box_width + "┐" + " " + "┌" + "─"*box_width + "┐")
        print("  " + "│" + f"  ?    ".center(box_width) + "│" + " " + "│" + f"  {bits_after[1]}    ".center(box_width) + "│" + " " + "│" + f"  {bits_after[2]}    ".center(box_width) + "│")
        print("  " + "└" + "─"*box_width + "┘" + " " + "└" + "─"*box_width + "┘" + " " + "└" + "─"*box_width + "┘")
        print()

        time.sleep(0.5)

        # XOR feedback
        if output_bit == 1:
            print("⚡ Output was 1, so XOR with polynomial!")
            print(f"   Polynomial {hex(polynomial)} = {bin(polynomial)}")

            state = state_after_shift ^ polynomial

            print(f"   {bin(state_after_shift):>8} (after shift)")
            print(f" ⊕ {bin(polynomial):>8} (polynomial)")
            print(f"   {'─'*8}")
            print(f"   {bin(state):>8} (new state)")
        else:
            print("✋ Output was 0, so NO XOR")
            state = state_after_shift

        print()

        # Show final state
        final_bits = [(state >> i) & 1 for i in range(n_bits-1, -1, -1)]
        print("  Final state for this step:")
        print("  " + "┌" + "─"*box_width + "┐" + " " + "┌" + "─"*box_width + "┐" + " " + "┌" + "─"*box_width + "┐")
        print("  " + "│" + f"  {final_bits[0]}    ".center(box_width) + "│" + " " + "│" + f"  {final_bits[1]}    ".center(box_width) + "│" + " " + "│" + f"  {final_bits[2]}    ".center(box_width) + "│")
        print("  " + "└" + "─"*box_width + "┘" + " " + "└" + "─"*box_width + "┘" + " " + "└" + "─"*box_width + "┘")
        print()

        if step < steps - 1:
            input("Press Enter for next step...")
            print("\n")

    print("="*70)
    print("COMPLETE!")
    print("="*70)
    print()
    print(f"Generated sequence: {sequence}")
    print(f"Length: {len(sequence)}")
    print()
    print("Notice: This sequence will repeat after a full cycle!")
    print(f"For a {n_bits}-bit LFSR, maximum period is 2^{n_bits} - 1 = {(1 << n_bits) - 1}")

def simple_comparison():
    """Show the difference between random and PRBS"""
    import random

    print("\n" + "="*70)
    print("COMPARISON: True Random vs PRBS")
    print("="*70)
    print()

    print("🎲 True Random (flipping a coin):")
    print("   Run 1:", [random.randint(0, 1) for _ in range(10)])
    print("   Run 2:", [random.randint(0, 1) for _ in range(10)])
    print("   ↑ Different every time!")
    print()

    print("🔧 PRBS (using our machine):")

    def mini_prbs():
        state = 0b101
        poly = 0x3
        result = []
        for _ in range(10):
            result.append(state & 1)
            state >>= 1
            if result[-1]:
                state ^= poly
        return result

    print("   Run 1:", mini_prbs())
    print("   Run 2:", mini_prbs())
    print("   ↑ Identical every time!")
    print()

    print("💡 KEY INSIGHT:")
    print("   - Random: Unpredictable, unreproducible")
    print("   - PRBS: Predictable, reproducible, but LOOKS random!")
    print()

def explain_xor():
    """Visual explanation of XOR"""
    print("\n" + "="*70)
    print("UNDERSTANDING XOR (⊕)")
    print("="*70)
    print()

    print("XOR means 'exclusive OR' - output 1 if inputs are DIFFERENT:")
    print()

    examples = [
        (0, 0, 0, "same → 0"),
        (0, 1, 1, "different → 1"),
        (1, 0, 1, "different → 1"),
        (1, 1, 0, "same → 0"),
    ]

    for a, b, result, explanation in examples:
        print(f"   {a} ⊕ {b} = {result}  ({explanation})")

    print()
    print("Think of it as 'disagreement detector':")
    print("   - If bits agree (both 0 or both 1) → output 0")
    print("   - If bits disagree (one 0, one 1) → output 1")
    print()

    print("Why XOR is perfect for PRBS:")
    print("   ✓ Simple to compute (just one operation)")
    print("   ✓ Creates complex mixing")
    print("   ✓ Reversible (XOR same value twice = original)")
    print("   ✓ No information lost")
    print()

def main():
    """Run the complete demonstration"""
    print()
    print_slowly("╔═══════════════════════════════════════════════════════════════════╗")
    print_slowly("║                                                                   ║")
    print_slowly("║         PRBS GENERATOR - INTERACTIVE VISUAL GUIDE                 ║")
    print_slowly("║         See Exactly How Fake Randomness Works!                    ║")
    print_slowly("║                                                                   ║")
    print_slowly("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    print("This demonstration will show you:")
    print("  1. How the shift register works")
    print("  2. When and why XOR happens")
    print("  3. How the sequence is generated")
    print()

    input("Press Enter to begin...")
    print()

    # First, explain XOR
    explain_xor()
    input("Press Enter to continue...")

    # Show the LFSR in action
    visualize_lfsr(n_bits=3, polynomial=0x3, initial_state=0b101, steps=8)

    # Show comparison
    simple_comparison()

    print("="*70)
    print("🎓 WHAT YOU LEARNED:")
    print("="*70)
    print()
    print("1. PRBS uses a shift register (boxes that shift right)")
    print("2. XOR feedback creates complex patterns")
    print("3. Output bit comes from rightmost position")
    print("4. Same starting state → same sequence (reproducible!)")
    print("5. Looks random but is completely deterministic")
    print()
    print("This is the foundation of:")
    print("  • GPS signals")
    print("  • Cell phone communication")
    print("  • WiFi")
    print("  • Touch screens")
    print("  • Cryptography")
    print()
    print("🎉 You now understand PRBS generation!")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Thanks for watching!")
        sys.exit(0)
