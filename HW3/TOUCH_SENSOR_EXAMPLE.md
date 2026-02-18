# Complete Touch Sensor Example - PRBS in Action
## A Step-by-Step Journey From Physics to Detection

---

## Part 1: The Physical Setup

### What We're Building

Imagine a touch sensor with **3 drive lines** and **1 sense line**:

```
         Sense Line (what we measure)
    ═════════════════════════════════════
              ║        ║        ║
              ║        ║        ║   ← Capacitive coupling
              ║        ║        ║
    ─────   ─────   ─────   (Drive lines)
    Drive1  Drive2  Drive3

    Position: 1cm     2cm     3cm
```

### The Physics

Each drive-sense pair acts like a **capacitor** (two plates with air gap):

```
Drive line (bottom plate) ───────────
                           ≈≈≈≈≈≈≈≈≈≈  ← Electric field
Sense line (top plate)  ═════════════

Capacitance = how much the field couples through
```

**Without finger:**
- Small baseline capacitance (electric field through air)

**With finger:**
- Finger absorbs/redirects field
- Capacitance changes (usually decreases)

---

## Part 2: The Problem - Why We Need PRBS

### The Slow Way (Traditional)

```
Step 1: Drive1 = ON,  Drive2 = OFF, Drive3 = OFF  → Measure sense
Step 2: Drive1 = OFF, Drive2 = ON,  Drive3 = OFF  → Measure sense
Step 3: Drive1 = OFF, Drive2 = OFF, Drive3 = ON   → Measure sense

Time required: 3 time slots
```

**For a phone screen with 100 lines: 100 time slots!**

### The Fast Way (CDMA with PRBS)

```
All drives ON simultaneously with different codes!
Drive1 = Code A
Drive2 = Code B  } All at once!
Drive3 = Code C

Time required: 1 time slot
```

**How do we separate them? CORRELATION!**

---

## Part 3: Creating the Codes

### Generate PRBS7 (7-bit sequence)

Let's use a simple 3-bit LFSR to generate a 7-bit sequence.

**Setup:**
- 3-bit LFSR
- Polynomial 0x6 (positions 1 and 2)
- Initial state: [0, 0, 1]

**Generation (see WORKED_EXAMPLE.md for details):**
```
PRBS7 sequence: [1, 1, 1, 0, 1, 0, 0]
Convert to bipolar: [+1, +1, +1, -1, +1, -1, -1]
```

### Assign Different Phases to Each Drive

```
Drive 1: [+1, +1, +1, -1, +1, -1, -1]  ← Phase offset 0
Drive 2: [+1, -1, +1, -1, -1, +1, +1]  ← Phase offset 2
Drive 3: [-1, +1, -1, -1, +1, +1, +1]  ← Phase offset 5
```

These are the **same** sequence, just **shifted circularly**!

**Why this works:** Remember autocorrelation? PRBS correlated with shifted version ≈ 0!

---

## Part 4: No Touch Scenario - Measuring Baseline

### The Capacitances (No Finger)

```
C1 = 10 pF  (Drive1 to Sense)
C2 = 8 pF   (Drive2 to Sense)
C3 = 12 pF  (Drive3 to Sense)
```

### Step-by-Step Signal Generation

**Time slot 0:**
```
Drive1 sends: +1 × 5V = +5V
Drive2 sends: +1 × 5V = +5V
Drive3 sends: -1 × 5V = -5V

What appears on sense line:
From Drive1: +5V × C1/(C1+C2+C3) = +5V × 10/30 = +1.67V
From Drive2: +5V × C2/(C1+C2+C3) = +5V × 8/30  = +1.33V
From Drive3: -5V × C3/(C1+C2+C3) = -5V × 12/30 = -2.00V

Total sense voltage: +1.67 + 1.33 - 2.00 = +1.00V
```

**Let me simplify this - use direct capacitive coupling:**

Actually, let's make this simpler. Each drive line voltage causes a proportional current/voltage on the sense line based on its capacitance:

```
Sense signal = C1×Drive1 + C2×Drive2 + C3×Drive3
```

### The Full Sequence (7 time samples)

Let me use simpler, normalized units where each drive sends its code directly:

```
Time:   0    1    2    3    4    5    6

Drive1: +1   +1   +1   -1   +1   -1   -1
Drive2: +1   -1   +1   -1   -1   +1   +1
Drive3: -1   +1   -1   -1   +1   +1   +1

Sense = 10×Drive1 + 8×Drive2 + 12×Drive3
(capacitance × code)
```

**Let's calculate each time sample:**

**t=0:**
```
Sense[0] = 10×(+1) + 8×(+1) + 12×(-1)
        = 10 + 8 - 12
        = 6
```

**t=1:**
```
Sense[1] = 10×(+1) + 8×(-1) + 12×(+1)
        = 10 - 8 + 12
        = 14
```

**t=2:**
```
Sense[2] = 10×(+1) + 8×(+1) + 12×(-1)
        = 10 + 8 - 12
        = 6
```

**t=3:**
```
Sense[3] = 10×(-1) + 8×(-1) + 12×(-1)
        = -10 - 8 - 12
        = -30
```

**t=4:**
```
Sense[4] = 10×(+1) + 8×(-1) + 12×(+1)
        = 10 - 8 + 12
        = 14
```

**t=5:**
```
Sense[5] = 10×(-1) + 8×(+1) + 12×(+1)
        = -10 + 8 + 12
        = 10
```

**t=6:**
```
Sense[6] = 10×(-1) + 8×(+1) + 12×(+1)
        = -10 + 8 + 12
        = 10
```

### Measured Sense Signal (No Touch)

```
Sense = [6, 14, 6, -30, 14, 10, 10]
```

This is what the ADC measures - a mixed signal from all 3 drives!

---

## Part 5: Extracting Capacitances with Correlation

### Recovering Drive1's Capacitance

**Correlate sense signal with Drive1's code:**

```
Sense:       [6,  14,   6, -30,  14,  10,  10]
Drive1 code: [+1, +1,  +1,  -1,  +1,  -1,  -1]

Multiply position by position:
  6 × +1 =  +6
 14 × +1 = +14
  6 × +1 =  +6
-30 × -1 = +30
 14 × +1 = +14
 10 × -1 = -10
 10 × -1 = -10

Sum = 6 + 14 + 6 + 30 + 14 - 10 - 10 = 50
```

**Wait, we should get 70 (10×7). Let me recalculate...**

Actually, I need to think about this more carefully. The issue is cross-correlation isn't zero.

Let me use a proper example with orthogonal codes:

---

## Part 5 (REVISED): Using Proper Orthogonal Codes

Let's use **Walsh codes** which are perfectly orthogonal:

```
Drive1 code: [+1, +1, +1, +1]  (Walsh 0)
Drive2 code: [+1, -1, +1, -1]  (Walsh 1)
Drive3 code: [+1, +1, -1, -1]  (Walsh 2)

Length N = 4 (shorter for clarity)
```

**Check orthogonality:**
```
Drive1 · Drive2 = 1×1 + 1×(-1) + 1×1 + 1×(-1) = 0 ✓
Drive1 · Drive3 = 1×1 + 1×1 + 1×(-1) + 1×(-1) = 0 ✓
Drive2 · Drive3 = 1×1 + (-1)×1 + 1×(-1) + (-1)×(-1) = 0 ✓
```

Perfect! They don't interfere.

### The Full Sequence (4 time samples)

**Capacitances:**
```
C1 = 10 pF
C2 = 8 pF
C3 = 12 pF
```

**Signals over time:**
```
Time:    0    1    2    3

Drive1: +1   +1   +1   +1
Drive2: +1   -1   +1   -1
Drive3: +1   +1   -1   -1
```

**Measured sense signal:**
```
t=0: 10×(+1) + 8×(+1) + 12×(+1) = 30
t=1: 10×(+1) + 8×(-1) + 12×(+1) = 14
t=2: 10×(+1) + 8×(+1) + 12×(-1) = 6
t=3: 10×(+1) + 8×(-1) + 12×(-1) = -10

Sense = [30, 14, 6, -10]
```

### Recover Drive1 Capacitance

**Correlate with Drive1's code:**
```
Sense:       [30,  14,   6, -10]
Drive1 code: [+1,  +1,  +1,  +1]

Multiply:
 30 × +1 = +30
 14 × +1 = +14
  6 × +1 =  +6
-10 × +1 = -10

Sum = 30 + 14 + 6 - 10 = 40
Divide by N: 40 ÷ 4 = 10 ✓

Recovered capacitance: 10 pF (exactly right!)
```

### Recover Drive2 Capacitance

**Correlate with Drive2's code:**
```
Sense:       [30,  14,   6, -10]
Drive2 code: [+1,  -1,  +1,  -1]

Multiply:
 30 × +1 = +30
 14 × -1 = -14
  6 × +1 =  +6
-10 × -1 = +10

Sum = 30 - 14 + 6 + 10 = 32
Divide by N: 32 ÷ 4 = 8 ✓

Recovered capacitance: 8 pF (exactly right!)
```

### Recover Drive3 Capacitance

**Correlate with Drive3's code:**
```
Sense:       [30,  14,   6, -10]
Drive3 code: [+1,  +1,  -1,  -1]

Multiply:
 30 × +1 = +30
 14 × +1 = +14
  6 × -1 =  -6
-10 × -1 = +10

Sum = 30 + 14 - 6 + 10 = 48
Divide by N: 48 ÷ 4 = 12 ✓

Recovered capacitance: 12 pF (exactly right!)
```

**Perfect separation! 🎉**

---

## Part 6: With Touch - Detecting Location

### A Finger Touches at Position 2cm

The finger changes capacitance:

```
Position:  1cm   2cm   3cm
Before:    10pF  8pF   12pF
After:     10pF  3pF   10pF  ← Position 2 dropped!
Change:     0   -5pF   -2pF
```

**Why:**
- Position 1 (1cm): No finger, no change
- Position 2 (2cm): Finger directly over, big change (-5pF)
- Position 3 (3cm): Finger partially affects, small change (-2pF)

### Measure With Touch

**New sense signal:**
```
t=0: 10×(+1) + 3×(+1) + 10×(+1) = 23  (was 30)
t=1: 10×(+1) + 3×(-1) + 10×(+1) = 17  (was 14)
t=2: 10×(+1) + 3×(+1) + 10×(-1) = 3   (was 6)
t=3: 10×(+1) + 3×(-1) + 10×(-1) = -3  (was -10)

Sense_touch = [23, 17, 3, -3]
```

### Extract New Capacitances

**Drive1:**
```
[23, 17, 3, -3] · [+1, +1, +1, +1]
= 23 + 17 + 3 - 3 = 40
÷ 4 = 10 pF (no change) ✓
```

**Drive2:**
```
[23, 17, 3, -3] · [+1, -1, +1, -1]
= 23 - 17 + 3 + 3 = 12
÷ 4 = 3 pF (dropped from 8!) ✓
```

**Drive3:**
```
[23, 17, 3, -3] · [+1, +1, -1, -1]
= 23 + 17 - 3 + 3 = 40
÷ 4 = 10 pF (dropped from 12) ✓
```

### Compare and Locate Touch

```
Position:  1cm    2cm    3cm
No touch:  10pF   8pF    12pF
Touch:     10pF   3pF    10pF
Difference: 0    -5pF   -2pF
            ↓      ↓      ↓
          none  TOUCH  slight
```

**Conclusion: Finger is at position 2cm!**

---

## Part 7: Adding Noise - The Real World

### Real Measurements Have Noise

Let's add Gaussian noise (σ = 2) to our sense measurements:

**Touch measurement with noise:**
```
True:  [23,   17,    3,   -3]
Noise: [1.2, -0.8,  1.5, -0.5]
Measured: [24.2, 16.2, 4.5, -3.5]
```

**Can we still recover the capacitances?**

### Extract Drive2 Capacitance (with noise)

```
Noisy sense: [24.2, 16.2, 4.5, -3.5]
Drive2 code: [+1,   -1,   +1,  -1]

Correlation:
24.2 × +1 = +24.2
16.2 × -1 = -16.2
 4.5 × +1 =  +4.5
-3.5 × -1 =  +3.5

Sum = 24.2 - 16.2 + 4.5 + 3.5 = 16.0
÷ 4 = 4.0 pF

True value: 3 pF
Error: 1 pF (due to noise)
```

Not bad! With longer sequences (PRBS511), the error would be much smaller due to **processing gain**.

### Processing Gain Calculation

**For our example (N=4):**
```
Processing gain = √4 = 2

Original SNR: signal/noise = 3/2 = 1.5
After correlation: SNR ≈ 1.5 × 2 = 3.0

Improvement: 6 dB
```

**For real system (N=511):**
```
Processing gain = √511 ≈ 22.6

Improvement: 27 dB! 🚀
```

---

## Part 8: Complete Python Simulation

Let me create a working simulation you can run:

```python
import numpy as np

# Setup
N = 4  # Code length
num_drives = 3

# Walsh codes (perfectly orthogonal)
codes = np.array([
    [+1, +1, +1, +1],  # Drive 1
    [+1, -1, +1, -1],  # Drive 2
    [+1, +1, -1, -1],  # Drive 3
])

# Capacitances (pF)
cap_notouch = np.array([10, 8, 12])
cap_touch = np.array([10, 3, 10])  # Touch at position 2

print("="*60)
print("CAPACITIVE TOUCH SENSOR SIMULATION")
print("="*60)

# Generate sense signal (no touch)
sense_notouch = codes.T @ cap_notouch  # Matrix multiply
print(f"\nSense (no touch): {sense_notouch}")

# Generate sense signal (with touch)
sense_touch = codes.T @ cap_touch
print(f"Sense (touch):    {sense_touch}")

# Recover capacitances by correlation
print("\n" + "="*60)
print("RECOVERING CAPACITANCES")
print("="*60)

for i in range(num_drives):
    # No touch
    corr_notouch = np.dot(sense_notouch, codes[i]) / N
    print(f"\nDrive {i+1} (no touch):")
    print(f"  True: {cap_notouch[i]} pF")
    print(f"  Recovered: {corr_notouch:.1f} pF")

    # With touch
    corr_touch = np.dot(sense_touch, codes[i]) / N
    print(f"  With touch: {corr_touch:.1f} pF")
    print(f"  Change: {corr_touch - corr_notouch:.1f} pF")

# Detect touch location
print("\n" + "="*60)
print("TOUCH DETECTION")
print("="*60)

differences = []
for i in range(num_drives):
    corr_notouch = np.dot(sense_notouch, codes[i]) / N
    corr_touch = np.dot(sense_touch, codes[i]) / N
    diff = corr_touch - corr_notouch
    differences.append(diff)
    print(f"Position {i+1}: Change = {diff:.1f} pF")

max_change_idx = np.argmin(differences)  # Most negative
print(f"\n→ Touch detected at position {max_change_idx + 1}!")

# Add noise and test
print("\n" + "="*60)
print("WITH NOISE (σ = 2)")
print("="*60)

noise = np.random.randn(N) * 2
sense_noisy = sense_touch + noise
print(f"\nNoisy sense: {sense_noisy}")

for i in range(num_drives):
    corr_noisy = np.dot(sense_noisy, codes[i]) / N
    corr_true = np.dot(sense_touch, codes[i]) / N
    print(f"\nDrive {i+1}:")
    print(f"  True capacitance: {corr_true:.1f} pF")
    print(f"  Noisy measurement: {corr_noisy:.1f} pF")
    print(f"  Error: {abs(corr_noisy - corr_true):.1f} pF")
```

Save this as `touch_sensor_demo.py` and run it!

---

## Part 9: Scaling to Real Systems

### Your Smartphone Touch Screen

**Real parameters:**
- **Drive lines:** 20-40
- **Sense lines:** 10-30
- **Total sensors:** 200-1200
- **Code length:** 127-511 bits
- **Scan rate:** 120-240 Hz
- **ADC resolution:** 12-16 bits

**Why CDMA is essential:**
```
Without CDMA: Scan 1200 positions sequentially
  → 1200 time slots → slow, can't keep up

With CDMA: Scan all drives simultaneously
  → Much fewer time slots → 120 Hz refresh!
```

### The Complete System

```
1. Transmitter:
   Each drive line gets unique code phase
   Drive with square waves (±5V)

2. Capacitive Coupling:
   Electric field couples through air/glass
   Finger changes the field

3. Receiver:
   ADC samples sense line
   512 samples at 10 MHz = 51.2 μs

4. Signal Processing:
   Correlate with each drive code
   Extract all capacitances
   Build 2D capacitance map

5. Touch Detection:
   Compare to baseline
   Find peaks in difference map
   Track multiple fingers
   Report to OS

Total time: ~8 ms → 120 Hz update rate
```

---

## Part 10: Key Takeaways

### What You Learned

1. **Physical principle:** Capacitive coupling between drive and sense

2. **The problem:** Need to scan many lines quickly

3. **The solution:** Transmit all codes simultaneously (CDMA)

4. **The magic:** Correlation separates mixed signals

5. **Noise resistance:** Processing gain = √N helps fight noise

6. **Real implementation:** Your phone does this thousands of times per second!

### Why It Works

```
Different codes → Nearly orthogonal
Orthogonal codes → Correlation separates them
Long codes → Processing gain fights noise
Fast ADC → Can sample at high rate
Parallel transmission → Fast scan
```

### The Math in Action

**Transmit:**
```
Mixed signal = C₁×Code₁ + C₂×Code₂ + C₃×Code₃
```

**Receive:**
```
Correlation with Codeᵢ extracts Cᵢ
All other terms → ~0 (orthogonal)
Noise → averaged out (processing gain)
```

**Result:**
```
Fast, accurate, multi-touch detection!
```

---

## Try It Yourself

### Run the Simulation

```bash
cd /Users/fengruizuo/Documents/EE217/HW3
source HW3/bin/activate
python touch_sensor_demo.py
```

### Experiments

**1. Change capacitances:**
```python
cap_touch = np.array([10, 5, 10])  # Different touch strength
```

**2. Add more noise:**
```python
noise = np.random.randn(N) * 5  # σ = 5 instead of 2
```

**3. Try more drives:**
```python
# Add Walsh 4-7 codes for 7 drive lines
```

**4. Longer codes:**
```python
# Use 8-bit Walsh codes (256 drives!)
```

### Questions to Think About

1. **Why does correlation work?**
   → Codes are orthogonal, dot product = 0

2. **Why is processing gain √N?**
   → Signal adds coherently (N), noise adds randomly (√N)

3. **What if codes aren't orthogonal?**
   → Interference (crosstalk) between channels

4. **Why use PRBS instead of Walsh?**
   → PRBS has better autocorrelation, easier to generate

5. **Could we use random codes?**
   → No! Receiver wouldn't know what to correlate with

---

## Summary

You've just seen **exactly** how your phone's touch screen works:

```
1. Generate orthogonal codes (PRBS phases)
2. Drive all lines simultaneously
3. Measure mixed sense signal
4. Correlate to separate each drive
5. Compare to baseline → find touch
6. Repeat 120 times per second
```

**It's not magic - it's math! 🎓**

The same principles power:
- Touch screens (this example)
- GPS satellites (different Gold codes)
- Cell phones (different spreading codes)
- WiFi (orthogonal subcarriers)

Understanding this one example gives you insight into dozens of modern technologies!
