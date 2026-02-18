# Complete Worked Example - PRBS and Correlation
## Step-by-Step Calculation You Can Follow Along

---

## Example 1: Generating a 7-bit PRBS by Hand

Let's generate PRBS7 step-by-step so you can see EXACTLY how it works.

### Setup
- **Length**: 3 bits (for simplicity, real PRBS7 is 7 bits)
- **Polynomial**: XOR positions 0 and 1
- **Initial state**: `[1, 0, 1]` (positions 2, 1, 0)

### The Rule (Galois LFSR)
```
1. Output the rightmost bit (position 0)
2. Shift everything right
3. If the output was 1, XOR the polynomial into the register
```

### Step-by-Step Execution

**Step 1:**
```
Current state: [1, 0, 1]
               pos2 pos1 pos0

Output: 1 (from position 0)
Shift right: [?, 1, 0]
Output was 1, so XOR with polynomial 0b11 (positions 0 and 1):
  [0, 1, 0]  (after shift)
⊕ [0, 1, 1]  (polynomial)
= [0, 0, 1]  (new state)

Sequence so far: [1]
```

**Step 2:**
```
Current state: [0, 0, 1]

Output: 1
Shift right: [?, 0, 0]
Output was 1, so XOR:
  [0, 0, 0]
⊕ [0, 1, 1]
= [0, 1, 1]

Sequence so far: [1, 1]
```

**Step 3:**
```
Current state: [0, 1, 1]

Output: 1
Shift right: [?, 0, 1]
Output was 1, so XOR:
  [0, 0, 1]
⊕ [0, 1, 1]
= [0, 1, 0]

Sequence so far: [1, 1, 1]
```

**Step 4:**
```
Current state: [0, 1, 0]

Output: 0
Shift right: [?, 0, 1]
Output was 0, so NO XOR:
= [0, 0, 1]

Sequence so far: [1, 1, 1, 0]
```

**Step 5:**
```
Current state: [0, 0, 1]

This is same as Step 2! The sequence repeats.

Final sequence: [1, 1, 1, 0, 1, 1, 1, 0, ...]
Period = 4  (but we skipped state [0,0,0])
```

---

## Example 2: Computing Autocorrelation by Hand

Let's compute the autocorrelation of a 5-bit sequence.

### The Sequence
```
Binary: [1, 0, 1, 1, 0]
Bipolar: [+1, -1, +1, +1, -1]
```

### Shift 0 (No shift)
```
Position:     0    1    2    3    4
Original:   [+1, -1,  +1,  +1, -1]
Shifted:    [+1, -1,  +1,  +1, -1] ← same

Multiply each position:
   +1 × +1 = +1
   -1 × -1 = +1
   +1 × +1 = +1
   +1 × +1 = +1
   -1 × -1 = +1

Sum = +1 +1 +1 +1 +1 = +5 ← Maximum!
```

### Shift 1
```
Position:     0    1    2    3    4
Original:   [+1, -1,  +1,  +1, -1]
Shifted:    [-1,  +1, -1,  +1,  +1] ← rotated left by 1
                                        (wraps around)

Multiply:
   +1 × -1 = -1
   -1 × +1 = -1
   +1 × -1 = -1
   +1 × +1 = +1
   -1 × +1 = -1

Sum = -1 -1 -1 +1 -1 = -3
```

### Shift 2
```
Original:   [+1, -1,  +1,  +1, -1]
Shifted:    [+1,  +1, -1,  +1, -1]

Multiply:
   +1 × +1 = +1
   -1 × +1 = -1
   +1 × -1 = -1
   +1 × +1 = +1
   -1 × -1 = +1

Sum = +1 -1 -1 +1 +1 = +1
```

### Shift 3
```
Original:   [+1, -1,  +1,  +1, -1]
Shifted:    [-1,  +1, +1,  -1,  +1]

Multiply:
   +1 × -1 = -1
   -1 × +1 = -1
   +1 × +1 = +1
   +1 × -1 = -1
   -1 × +1 = -1

Sum = -1 -1 +1 -1 -1 = -3
```

### Shift 4
```
Original:   [+1, -1,  +1,  +1, -1]
Shifted:    [+1, -1,  +1,  +1, -1]

This is back to shift 0!

Sum = +5
```

### Autocorrelation Results
```
Shift:  0    1    2    3    4
Corr:   +5   -3   +1   -3   +5
        ↑                   ↑
      peak                peak
                            (repeats)
```

**Plot:**
```
  +5 |  *              *
     |
   0 |_____________________
     |
  -3 |     *        *
     +------------------
        0  1  2  3  4
```

Not a perfect PRBS (would need all -1 except peak), but you can see the principle!

---

## Example 3: CDMA Signal Separation

Let's separate 2 users' signals using correlation.

### User Codes (both length 5)
```
User A code: [+1, -1, +1, +1, -1]
User B code: [-1, +1, +1, -1, +1]  (different pattern)
```

### Users Transmit
```
User A sends data value: 3
User B sends data value: -2
```

### Create Transmitted Signals
```
User A signal = 3 × [+1, -1, +1, +1, -1]
              = [+3, -3, +3, +3, -3]

User B signal = -2 × [-1, +1, +1, -1, +1]
              = [+2, -2, -2, +2, -2]
```

### Combined Signal (what receiver gets)
```
Position:  0    1    2    3    4
User A:   +3   -3   +3   +3   -3
User B:   +2   -2   -2   +2   -2
         ─────────────────────────
Total:    +5   -5   +1   +5   -5
```

This mixed signal arrives at the receiver. How do we separate them?

### Recover User A's Data

Correlate received signal with User A's code:
```
Received:      [+5, -5, +1, +5, -5]
User A code:   [+1, -1, +1, +1, -1]

Multiply position by position:
   +5 × +1 = +5
   -5 × -1 = +5
   +1 × +1 = +1
   +5 × +1 = +5
   -5 × -1 = +5

Sum = +5 +5 +1 +5 +5 = +21
Normalize = 21 ÷ 5 = 4.2 ≈ 4
```

Hmm, we sent 3 but got 4. Let me recalculate...

Actually, let me recalculate the correlation properly:
```
Sum = +21
Code length = 5
Code squared sum = 1² + 1² + 1² + 1² + 1² = 5

Estimated amplitude = 21/5 = 4.2
```

Wait, this should give us 3. Let me think about why...

Actually, the issue is cross-correlation! Let me compute it:

### Cross-correlation check
```
User A code: [+1, -1, +1, +1, -1]
User B code: [-1, +1, +1, -1, +1]

Multiply:
   +1 × -1 = -1
   -1 × +1 = -1
   +1 × +1 = +1
   +1 × -1 = -1
   -1 × +1 = -1

Sum = -1 -1 +1 -1 -1 = -3
```

So the codes aren't perfectly orthogonal! Cross-correlation is -3, not 0.

### Correct Recovery

When we correlate received with User A's code:
```
User A contribution: 3 × 5 = 15  (signal × code length)
User B contribution: -2 × (-3) = 6  (interferer × cross-corr)

Total correlation = 15 + 6 = 21
Estimated A = 21/5 = 4.2 (has error due to non-zero cross-corr)
```

**For perfect separation, we need codes with zero cross-correlation!**

This is why real CDMA uses carefully designed codes (Gold codes, Walsh codes) that are truly orthogonal.

---

## Example 4: Processing Gain Calculation

Let's see how correlation fights noise.

### Signal Without Noise
```
PRBS code: [+1, -1, +1, -1, +1, -1, +1]  (length 7)
Data to send: 2
Transmitted: [+2, -2, +2, -2, +2, -2, +2]
```

### Received With Noise
```
Let's say Gaussian noise with σ = 1 is added:

Sample 0: +2 + 0.5 = +2.5
Sample 1: -2 - 0.8 = -2.8
Sample 2: +2 + 1.2 = +3.2
Sample 3: -2 - 0.3 = -2.3
Sample 4: +2 + 0.1 = +2.1
Sample 5: -2 + 0.9 = -1.1
Sample 6: +2 - 0.5 = +1.5

Received: [+2.5, -2.8, +3.2, -2.3, +2.1, -1.1, +1.5]
```

Looking at this noisy signal, it's hard to tell what the data value was!

### Correlate to Detect
```
Received:    [+2.5, -2.8, +3.2, -2.3, +2.1, -1.1, +1.5]
PRBS code:   [+1,   -1,   +1,   -1,   +1,   -1,   +1  ]

Multiply:
   +2.5 × +1 = +2.5
   -2.8 × -1 = +2.8
   +3.2 × +1 = +3.2
   -2.3 × -1 = +2.3
   +2.1 × +1 = +2.1
   -1.1 × -1 = +1.1
   +1.5 × +1 = +1.5

Sum = 2.5 + 2.8 + 3.2 + 2.3 + 2.1 + 1.1 + 1.5 = 15.5
Divide by length: 15.5 ÷ 7 = 2.21 ≈ 2 ✓
```

**We recovered the data value of 2!**

### Processing Gain
```
Original SNR (signal-to-noise):
   Signal amplitude: 2
   Noise std dev: 1
   SNR = 2/1 = 2 (6 dB)

After correlation:
   Processing gain = √7 = 2.65
   Effective SNR = 2 × 2.65 = 5.3 (14.5 dB)

Improvement: 14.5 - 6 = 8.5 dB
```

The noise in each sample was significant (up to 50% of signal), but correlation averaged it out!

---

## Example 5: DFT Calculation (4-point)

Let's compute a 4-point DFT by hand.

### Input Signal
```
x[n] = [1, 2, 3, 4]  (just some numbers)
```

### DFT Formula
```
X[k] = Σ(n=0 to N-1) x[n] × e^(-j2πkn/N)

For N=4:
e^(-j2πkn/4) = cos(-2πkn/4) + j×sin(-2πkn/4)
```

### Compute Each Frequency Bin

**X[0] - DC component:**
```
k = 0:
X[0] = x[0]×e^0 + x[1]×e^0 + x[2]×e^0 + x[3]×e^0
     = 1×1 + 2×1 + 3×1 + 4×1
     = 10
```

**X[1] - First frequency:**
```
k = 1:
e^(-j2π×0/4) = 1
e^(-j2π×1/4) = cos(-90°) + j×sin(-90°) = 0 - j = -j
e^(-j2π×2/4) = cos(-180°) + j×sin(-180°) = -1
e^(-j2π×3/4) = cos(-270°) + j×sin(-270°) = 0 + j = j

X[1] = 1×1 + 2×(-j) + 3×(-1) + 4×j
     = 1 - 2j - 3 + 4j
     = -2 + 2j
```

**X[2] - Second frequency:**
```
k = 2:
e^(-j2π×0/4) = 1
e^(-j2π×2/4) = -1
e^(-j2π×4/4) = 1
e^(-j2π×6/4) = -1

X[2] = 1×1 + 2×(-1) + 3×1 + 4×(-1)
     = 1 - 2 + 3 - 4
     = -2
```

**X[3] - Third frequency:**
```
k = 3:
X[3] = 1×1 + 2×j + 3×(-1) + 4×(-j)
     = 1 + 2j - 3 - 4j
     = -2 - 2j
```

### Result
```
X[0] = 10       (DC - average value)
X[1] = -2 + 2j  (first harmonic)
X[2] = -2       (second harmonic)
X[3] = -2 - 2j  (third harmonic)

Magnitudes:
|X[0]| = 10
|X[1]| = √(4+4) = 2.83
|X[2]| = 2
|X[3]| = 2.83
```

The signal has a strong DC component (10) and some oscillating parts.

---

## Example 6: Nyquist Sampling

### Problem
A music signal has frequencies from 20 Hz to 20 kHz. What sampling rate do we need?

### Solution
```
Bandwidth B = 20,000 Hz (highest frequency)
Nyquist rate = 2B = 2 × 20,000 = 40,000 Hz

Therefore: Sample at 40 kHz or higher
```

**Real systems:**
CD audio uses 44.1 kHz (safely above 40 kHz)

### What If We Sample Too Slow?

Sample at 30 kHz (below Nyquist):
```
A 18 kHz tone would be sampled correctly
A 19 kHz tone would be sampled correctly
A 20 kHz tone would ALIAS down to:
   30 - 20 = 10 kHz

Now it sounds like a 10 kHz tone instead!
```

**Aliasing causes high frequencies to appear as low frequencies.**

---

## Example 7: Frequency Resolution

### Problem
You want to distinguish between 440 Hz (A note) and 445 Hz (slightly sharp A). What's the minimum recording time?

### Solution
```
Frequency difference: 445 - 440 = 5 Hz
Need resolution: Δf ≤ 5 Hz

Formula: Δf = 1/T
Therefore: T ≥ 1/5 = 0.2 seconds

Record for at least 0.2 seconds (200 ms)
```

### Number of Samples
```
If sampling at 8 kHz:
N = fs × T = 8000 × 0.2 = 1600 samples

For FFT (power of 2):
2^10 = 1024 (too small)
2^11 = 2048 (good!)

Use 2048-point FFT
```

---

## Practice: Do These Yourself!

### Problem 1
Generate 5 steps of a 4-bit LFSR starting with [1,0,0,1] using polynomial 0x3 (XOR positions 0 and 1).

### Problem 2
Compute autocorrelation at shifts 0, 1, 2 for sequence [+1, +1, -1, +1].

### Problem 3
Signal has frequencies up to 5 kHz. You want 25 Hz resolution. Find:
- Minimum sampling rate
- Minimum recording time
- Minimum samples (power of 2)

---

## Answers to Practice Problems

### Answer 1
```
Polynomial 0x3 = 0b0011 (XOR positions 0 and 1)
Initial: [1,0,0,1]

Step 0: Output=1, Shift→[?,1,0,0], XOR→[0,1,0,0]⊕[0,0,1,1]=[0,1,1,1]
Step 1: Output=1, Shift→[?,0,1,1], XOR→[0,0,1,1]⊕[0,0,1,1]=[0,0,0,0]

Uh oh! Hit all zeros. This means 0x3 might not be a good polynomial for 4 bits.

Try polynomial 0x9 (positions 0 and 3):
... (you can work this out!)
```

### Answer 2
```
Sequence: [+1, +1, -1, +1]

Shift 0:
[+1,+1,-1,+1] · [+1,+1,-1,+1] = 1+1+1+1 = 4

Shift 1:
[+1,+1,-1,+1] · [+1,+1,+1,-1] = 1+1-1-1 = 0

Shift 2:
[+1,+1,-1,+1] · [-1,+1,+1,+1] = -1+1-1+1 = 0
```

### Answer 3
```
fs_min = 2 × 5000 = 10 kHz
T_min = 1/25 = 0.04 seconds (40 ms)
N_min = 10000 × 0.04 = 400 samples
Round to power of 2: 512 (2^9)
```

---

## Key Formulas Summary

**PRBS:**
- Period = 2^n - 1
- Autocorrelation peak = N
- Off-peak = -1 (for bipolar)

**Correlation:**
- R[k] = Σ x[i] × y[i+k]
- Processing gain = √N

**DFT:**
- X[k] = Σ x[n] × e^(-j2πkn/N)
- Inverse: x[n] = (1/N) Σ X[k] × e^(+j2πkn/N)

**Sampling:**
- fs ≥ 2B (Nyquist)
- Δf = 1/T (resolution)
- N = fs × T (samples)

**LTI System:**
- Y[k] = H(2πk/N) × X[k]

Now you can work through any problem step-by-step!
