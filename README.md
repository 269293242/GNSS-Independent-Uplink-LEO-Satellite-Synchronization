# GNSS-Independent Uplink LEO Satellite Synchronization

This repository provides a **reproduction and partial extension** of the paper:

> **"GNSS-Independent Uplink LEO Satellite Synchronization"**  
> Fredrik Berggren, Branislav M. Popović  
> IEEE Communications Letters, 2025

---

## Overview

This project reproduces the main simulation results of the paper, including:

- SPI-ZC sequence performance  
- Detection probability under large CFO  
- Comparison between different detection methods  

Some additional experiments and extended detection approaches are also included.

---

## Repository Structure

```
├── figure1_simulation.py   # Fig.1 reproduction (sequence comparison)
├── figure2_simulation.py   # Fig.2 reproduction (detector comparison)
├── my_first_test.py        # Extended simulation
├── my_second_test.py       # Improved detection methods
````

---

## How to Run

### Requirements

```bash
pip install numpy matplotlib pandas
```

### Run

```bash
python figure1_simulation.py
python figure2_simulation.py
```

---

## Notes

* The current implementation evaluates **timing detection only**
* Minor differences from the paper may occur due to simulation randomness

---

## Disclaimer

This repository is an independent reproduction of the referenced paper.

* All methods and theoretical concepts belong to the original authors
* This implementation is for academic and research purposes only

---

## Reference

```bibtex
@article{berggren2025gnss,
  title={GNSS-Independent Uplink LEO Satellite Synchronization},
  author={Berggren, Fredrik and Popovic, Branislav M.},
  journal={IEEE Communications Letters},
  year={2025}
}
```

---
