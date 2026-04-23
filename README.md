# GNSS-Independent Uplink LEO Satellite Synchronization

This repository provides a simplified reproduction of the paper:

> **GNSS-Independent Uplink LEO Satellite Synchronization**  
> Fredrik Berggren, Branislav M. Popović  
> *IEEE Communications Letters*, 2025

## Overview

This project reproduces the main simulation trends reported in the paper for **GNSS-independent uplink synchronization in LEO satellite systems**.

The current repository focuses on:

- **Fig. 1 reproduction**: comparison between SPI-ZC and Double-ZC sequences
- **Fig. 2 reproduction**: comparison of different timing detection methods for SPI-ZC
- Monte Carlo simulation of detection performance under different SNR conditions

This is a compact and code-oriented reproduction repository intended for academic study and further development.

## Repository Structure

```text
├── figure1_simulation.py
├── figure2_simulation.py
├── caculate.m
├── Fig.1 Probability of detection as function of SNR for N = 9.png
├── Fig. 2 Probability of timing detection as function of SNR for different detection methods of SPI-ZC sequence.png
├── README.md
└── LICENSE
````

### File Description

* `figure1_simulation.py`
  Reproduces the Fig. 1 style result: detection probability versus SNR for SPI-ZC and Double-ZC sequences.

* `figure2_simulation.py`
  Reproduces the Fig. 2 style result: timing detection probability versus SNR for different SPI-ZC detection methods.

* `caculate.m`
  Auxiliary MATLAB script kept in the repository for reference.

* `*.png`
  Example output figures corresponding to the reproduced simulation results.

## Requirements

Install the required Python packages:

```bash
pip install numpy matplotlib pandas
```

## How to Run

Run the two main Python scripts directly:

```bash
python figure1_simulation.py
python figure2_simulation.py
```

Each script will:

* run the Monte Carlo simulation,
* print numerical results in the terminal,
* display the corresponding figure.

## Notes

* This repository currently focuses on **timing detection performance**.
* The implementation is a **simplified reproduction**, intended to capture the main result trends rather than provide a full system-level reproduction of the paper.
* Minor differences from the published figures may appear due to modeling simplifications, parameter interpretation, and Monte Carlo randomness.

## Disclaimer

This repository is an independent academic reproduction of the referenced paper.

* All theoretical methods and original ideas belong to the authors of the paper.
* This code is provided for research, learning, and non-commercial academic use.

## Citation

If you use this repository, please cite the original paper:

```bibtex
@article{berggren2025gnss,
  title={GNSS-Independent Uplink LEO Satellite Synchronization},
  author={Berggren, Fredrik and Popović, Branislav M.},
  journal={IEEE Communications Letters},
  year={2025}
}
```

## License

This project is released under the MIT License.
