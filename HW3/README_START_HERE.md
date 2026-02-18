# 🎓 EE217 HW3 - Complete Learning Package
## Start Here! Your Guide to Everything

---

## 📋 Quick Start Guide

**If you just want to submit the homework:**
→ Open `HW3_Submission.txt` and paste into Google Docs

**If you want to understand the concepts:**
→ Read this file, then follow the learning path below

---

## 📦 What You Have

### For Submission ✅
1. **HW3_Submission.txt** - Ready to paste into Google Docs
   - All answers written in your casual tone
   - References to code and plots
   - Complete explanations

2. **hw3_solution.py** - Working Python code
   - Solves all 3 problems
   - Well-commented and educational
   - Generates all required plots

3. **All Plots Generated** (6 PNG files)
   - problem1b_autocorr_prbs511.png
   - problem1b_autocorr_255bit.png
   - problem1c_crosscorr.png
   - problem2_touch_sensor.png
   - problem3c_dtft.png
   - problem3c_ifft_comparison.png

### For Understanding 🧠

4. **BEGINNERS_GUIDE.md** ⭐ START HERE!
   - 14 chapters covering everything
   - Explained like you're a beginner
   - Tons of analogies and examples
   - No concept is too basic to explain
   - ~50 pages of detailed explanations

5. **CONCEPTS_EXPLAINED.md** (Advanced)
   - Technical deep-dive
   - Mathematical foundations
   - Real-world applications
   - Study tips and misconceptions

6. **WORKED_EXAMPLE.md** 📝 WORK THROUGH THIS!
   - 7 complete examples
   - Step-by-step calculations
   - Practice problems with answers
   - Hand-calculations you can follow

7. **interactive_demo.py** 🎮 RUN THIS!
   - 5 live demonstrations
   - Generates 4 visualization plots
   - See concepts in action
   - Experiment with parameters

8. **verify_1b.py** ✓ VERIFICATION
   - Double-checks Problem 1b answers
   - Shows all your work is correct

---

## 🎯 How To Use This Package

### Path 1: Quick Review (30 minutes)
**Goal:** Understand enough to explain your answers

```
1. Read BEGINNERS_GUIDE.md - Chapters 1-3 (PRBS basics)
2. Look at demo_autocorrelation.png
3. Read BEGINNERS_GUIDE.md - Chapter 5 (CDMA)
4. Look at demo_cdma.png
5. Skim your HW3_Submission.txt to see answers
```

### Path 2: Deep Understanding (3-4 hours)
**Goal:** Really learn this stuff

```
1. Read BEGINNERS_GUIDE.md completely (all 14 chapters)
2. Run: python interactive_demo.py
3. Study all 4 demo plots
4. Work through WORKED_EXAMPLE.md
5. Do the practice problems
6. Try modifying hw3_solution.py
```

### Path 3: Master Level (1-2 days)
**Goal:** Become an expert

```
1. Complete Path 2
2. Read CONCEPTS_EXPLAINED.md (advanced topics)
3. Implement LFSR from scratch without looking
4. Modify interactive_demo.py:
   - Try different PRBS polynomials
   - Change noise levels
   - Add more users to CDMA
5. Read about Gold codes and real GPS systems
6. Explain concepts to someone else
```

---

## 📚 Reading Order By Concept

### Understanding PRBS (Problem 1)
```
1. BEGINNERS_GUIDE.md - Chapter 1 (What is PRBS?)
2. BEGINNERS_GUIDE.md - Chapter 2 (How LFSRs work)
3. WORKED_EXAMPLE.md - Example 1 (Generate PRBS by hand)
4. Run interactive_demo.py to see LFSR steps
5. BEGINNERS_GUIDE.md - Chapter 3 (Autocorrelation)
6. WORKED_EXAMPLE.md - Example 2 (Calculate autocorr by hand)
7. BEGINNERS_GUIDE.md - Chapter 4 (Why subsequence fails)
```

### Understanding Touch Sensors (Problem 2)
```
1. BEGINNERS_GUIDE.md - Chapter 5 (CDMA cocktail party)
2. WORKED_EXAMPLE.md - Example 3 (CDMA separation)
3. Look at demo_cdma.png
4. BEGINNERS_GUIDE.md - Chapter 6 (Touch application)
5. BEGINNERS_GUIDE.md - Chapter 7 (Processing gain)
6. WORKED_EXAMPLE.md - Example 4 (Processing gain calc)
7. Look at demo_processing_gain.png
```

### Understanding DFT/FFT (Problem 3)
```
1. BEGINNERS_GUIDE.md - Chapter 8 (DFT basics)
2. WORKED_EXAMPLE.md - Example 5 (4-point DFT by hand)
3. BEGINNERS_GUIDE.md - Chapter 9 (Sampling theory)
4. WORKED_EXAMPLE.md - Examples 6-7 (Nyquist, resolution)
5. BEGINNERS_GUIDE.md - Chapter 10 (Time-domain aliasing)
6. Look at demo_time_aliasing.png
```

---

## 🎨 Visual Learning Resources

### Plots You Can Study

**Demo Plots (Conceptual):**
- `demo_autocorrelation.png` - See the spike at zero shift
- `demo_cdma.png` - Watch signals mix and separate
- `demo_time_aliasing.png` - Compare N=20 vs N=100
- `demo_processing_gain.png` - Signal recovery from noise

**Homework Plots (Your Answers):**
- `problem1b_autocorr_prbs511.png` - Perfect PRBS
- `problem1b_autocorr_255bit.png` - Imperfect subsequence
- `problem1c_crosscorr.png` - Two different generators
- `problem2_touch_sensor.png` - Touch location detection
- `problem3c_dtft.png` - Frequency domain
- `problem3c_ifft_comparison.png` - Aliasing demonstration

**Verification Plot:**
- `verify_problem1b.png` - Detailed analysis of Problem 1b

---

## 🔍 Quick Reference

### Key Concepts Explained

| Concept | What It Is | Why It Matters | Where to Learn |
|---------|-----------|----------------|----------------|
| **PRBS** | Deterministic "random" sequence | Synchronization, CDMA codes | Beginners Ch 1-2 |
| **LFSR** | Shift register with feedback | Generates PRBS | Beginners Ch 2, Worked Ex 1 |
| **Autocorrelation** | Compare signal with shifted self | Detect patterns, sync | Beginners Ch 3, Worked Ex 2 |
| **CDMA** | Multiple users, same frequency | Cell phones, GPS | Beginners Ch 5, Worked Ex 3 |
| **Processing Gain** | Correlation fights noise | Weak signal detection | Beginners Ch 7, Worked Ex 4 |
| **DFT** | Time → Frequency | Spectral analysis | Beginners Ch 8, Worked Ex 5 |
| **Nyquist** | Sample at ≥ 2×bandwidth | Avoid aliasing | Beginners Ch 9, Worked Ex 6 |
| **Time Aliasing** | Undersample frequency domain | Understand sampling duality | Beginners Ch 10 |

### Key Formulas

```
PRBS Period:              2^n - 1
Autocorrelation Peak:     N (sequence length)
Processing Gain:          √N
Nyquist Rate:             fs ≥ 2B
Frequency Resolution:     Δf = 1/T
DFT:                      X[k] = Σ x[n]·e^(-j2πkn/N)
LTI Output:               Y[k] = H(2πk/N)·X[k]
```

### Common Mistakes to Avoid

❌ **Don't:**
- Start LFSR with all zeros (gets stuck!)
- Think PRBS is actually random (it's deterministic)
- Forget to convert to bipolar (+1/-1) for correlation
- Sample below Nyquist rate (causes aliasing)
- Use subsequence as true PRBS (broken feedback)

✅ **Do:**
- Use primitive polynomials for maximal length
- Normalize correlation by sequence length
- Check cross-correlation for CDMA codes
- Remember processing gain = √N
- Verify Nyquist and resolution requirements

---

## 💡 "I Still Don't Get..."

### "Why do we need pseudo-random?"
→ **BEGINNERS_GUIDE.md - Chapter 1, first 3 sections**
TL;DR: Real random can't be reproduced. Both sender and receiver need the same "random" sequence.

### "How does correlation separate signals?"
→ **BEGINNERS_GUIDE.md - Chapter 5 (cocktail party analogy)**
→ **WORKED_EXAMPLE.md - Example 3**
→ **Look at demo_cdma.png**

### "What's the difference between DFT and FFT?"
→ **BEGINNERS_GUIDE.md - Chapter 12, Confusion 5**
TL;DR: Same result, FFT is just faster. Like division vs calculator.

### "Why is autocorrelation -1 not 0?"
→ **BEGINNERS_GUIDE.md - Chapter 12, Confusion 4**
→ **CONCEPTS_EXPLAINED.md - PRBS Properties section**

### "What causes time-domain aliasing?"
→ **BEGINNERS_GUIDE.md - Chapter 10**
→ **Look at demo_time_aliasing.png**
TL;DR: Not enough frequency samples, signal doesn't fit in one period.

---

## 🎮 Interactive Learning

### Run These Commands

```bash
# Activate virtual environment
source HW3/bin/activate

# See all demos
python interactive_demo.py

# Verify Problem 1b
python verify_1b.py

# Run full homework solution
python hw3_solution.py
```

### Experiments to Try

**Modify interactive_demo.py:**

1. **Change PRBS polynomial:**
   - Line 20: Try polynomial = 0x09 instead of 0x12
   - See how sequence changes

2. **Add noise:**
   - Line 180: Increase noise from 0.5 to 2.0
   - See processing gain work harder

3. **Change N for time aliasing:**
   - Line 220: Try N = [5, 10, 15, 25]
   - See aliasing get worse/better

4. **More CDMA users:**
   - Add User 4 and User 5
   - See if separation still works

---

## 🏆 Self-Assessment

### Can You Explain These to a Friend?

**Basic Level:**
- [ ] What PRBS is and why we use it
- [ ] How correlation works (pattern matching)
- [ ] Why CDMA doesn't cause interference
- [ ] What DFT does (time → frequency)
- [ ] Nyquist sampling rule

**Intermediate Level:**
- [ ] How LFSR generates sequences
- [ ] Why autocorrelation has a spike
- [ ] Processing gain formula and meaning
- [ ] Frequency resolution concept
- [ ] Why subsequence isn't true PRBS

**Advanced Level:**
- [ ] Primitive polynomials
- [ ] Cross-correlation properties
- [ ] Matched filtering optimality
- [ ] Time-frequency duality
- [ ] Real-world system design

### Can You Do These Calculations?

- [ ] Generate 10 bits of PRBS by hand
- [ ] Calculate autocorrelation for 5-bit sequence
- [ ] Separate 2 CDMA signals
- [ ] Compute 4-point DFT
- [ ] Determine sampling requirements

**If yes to all: You've mastered it! 🎉**

---

## 📞 Getting Help

### When You're Stuck

1. **Concept confusion:**
   → Read the relevant chapter in BEGINNERS_GUIDE.md
   → Look at the corresponding plot
   → Work through WORKED_EXAMPLE.md for that topic

2. **Math confusion:**
   → WORKED_EXAMPLE.md has step-by-step calculations
   → Follow along with your own numbers

3. **Code confusion:**
   → hw3_solution.py has detailed comments
   → interactive_demo.py shows concepts in action

4. **Still stuck:**
   → Read CONCEPTS_EXPLAINED.md (more technical)
   → Check the "Common Confusions" sections
   → Try teaching it to someone else (best way to learn!)

---

## 🎯 Your Homework Answers

### Problem 1: PRBS Signaling
✅ Generated PRBS7, 127, 511, 1023
✅ PRBS511 has perfect autocorrelation (511 at 0, -1 elsewhere)
✅ 255-bit subsequence is NOT a true PRBS255
   - Max off-peak: 27 (should be -1)
   - Depends on which subsequence
✅ Two generators (0x110 vs 0x108) are different
   - Max cross-correlation: 45

### Problem 2: Touch Sensors
✅ Found 5 correlation peaks (capacitances)
✅ Touch location: **10.86 mm**
   - Biggest change at 10mm position (-16.63)
   - Gaussian fit confirms location
✅ Noise per sample: 0.097

### Problem 3: DFT/FFT
✅ Y(k) = H(2πk/N) · X(k)
✅ Minimum requirements:
   - Sampling rate: 8 kHz
   - Record length: 20 ms
   - Samples: 256 (2^8)
✅ N=20 shows time-domain aliasing
   - N=100 works much better
   - Signal must decay within period

---

## 🚀 Next Steps

**After Understanding This Homework:**

1. **Related Topics to Explore:**
   - Gold codes and GPS systems
   - Walsh codes and OFDM
   - Spread spectrum communications
   - Matched filtering theory
   - Modern MIMO systems

2. **Practical Projects:**
   - Build a simple CDMA transceiver
   - Create a touch sensor with Arduino
   - Implement GPS signal acquisition
   - Design a software-defined radio

3. **Further Learning:**
   - Communications theory course
   - Digital signal processing
   - Information theory
   - Wireless systems design

**You now have the foundations for modern wireless systems! 🎓**

---

## 📝 Summary

You have:
- ✅ Complete homework submission ready
- ✅ Working code for all problems
- ✅ All required plots generated
- ✅ Beginner's guide (50+ pages)
- ✅ Advanced concepts explained
- ✅ 7 worked examples
- ✅ Interactive demonstrations
- ✅ Verification of answers

**Everything you need to:**
1. Submit your homework
2. Understand the concepts
3. Ace the exam
4. Apply to real projects

**Good luck! 🍀**
