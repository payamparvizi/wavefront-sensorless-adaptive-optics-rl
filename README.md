## Action-Regularized Reinforcement Learning for Adaptive Optics in Optical Satellite Communication

This repository contains the official implementation of **State-Adaptive Proportional Policy Smoothing (SAPPS)**, a policy regularization method designed to produce **smooth yet responsive control policies** in **reinforcement learning (RL) environment for wavefront sensorless adaptive optics (AO)**

SAPPS suppresses high-frequency oscillations in learned policies **without compromising performance**, particularly in **dynamic environments** where rapid adaptation is required.

📄 **Paper**: *Action-Regularized Reinforcement Learning for Adaptive Optics in Optical Satellite Communication*    
🔗 **Preprint**: [Optica Open](https://doi.org/10.1364/opticaopen.30043543)   
👤 **Authors**: [Payam Parvizi](https://www.linkedin.com/in/payamparvizi/), Colin Bellinger, Ross Cheriton, Abhishek Naik, Davide Spinello

The environment is designed to evaluate **policy smoothness and control responsiveness** in **highly dynamic optical systems**, with a primary motivation drawn from **satellite-to-ground optical communication** scenarios.

---

## Abstract

Optical satellite-to-ground communication enables terabit-scale data transmission through the atmosphere. However, atmospheric turbulence distorts the optical wavefront, significantly reducing the coupling efficiency into standard long-haul telecommunications fibers. While adaptive optics systems can correct these distortions, they are costly and complex. In this work, we introduce the State-Adaptive Proportional Policy Smoothing (SAPPS) method, a reinforcement learning-based approach for wavefront sensorless adaptive optics tailored for low-cost optical satellite communication. Our results demonstrate that SAPPS consistently maintains high coupling efficiency with low action fluctuation compared to baseline methods, enabling more stable and reliable operation under challenging atmospheric conditions.

---


## Method Overview

SAPPS is a general policy regularization technique that can be integrated into deep RL algorithms to improve policy smoothness in both static and dynamic continuous-control settings. Rather than directly penalizing action magnitude, SAPPS regularizes the change between consecutive actions based on the relative change between consecutive observations.

<p align="center">
  <img src="https://github.com/user-attachments/assets/3b4df3bc-2d37-49bb-ba33-a41ac0952c8d" align="center" width="400">
</p>

This adaptive formulation:
- penalizes unnecessary action fluctuations when state changes are small
- preserves responsiveness when large observation changes require rapid control adaptation

SAPPS is implemented within **Proximal Policy Optimization (PPO)** and compared against:
- Vanilla PPO  
- PPO with Conditioning for Action Policy Smoothness (CAPS)  
- Flat mirror scenario


## Overview

Wavefront sensorless adaptive optics aims to correct atmospheric wavefront distortions **without explicit wavefront sensing**, relying instead on low-dimensional photodetector measurements and closed-loop control of a deformable mirror.

<p align="center">
  <img src="https://github.com/user-attachments/assets/2b1b4781-e5b9-4d0d-91ef-a9da287a2d6c" align="center" width="400">
</p>


This environment formulates the AO control problem as a **continuous-control Markov Decision Process (MDP)** and is used to evaluate reinforcement learning algorithms under conditions where:

- observations are low-dimensional and partially observable,
- dynamics can change rapidly due to atmospheric turbulence, and
- excessive control oscillations directly degrade optical performance.

The focus of this environment is to assess whether **SAPPS** enables **smooth yet responsive control under** such challenging conditions.

---

## Key Results

The results below summarize fiber coupling efficiency and action smoothness for **PPO**, **CAPS**, and **SAPPS (proposed method)**, and for the **flat mirror** across adaptive optics scenarios with increasing atmospheric drift velocity.

At low drift velocity (5 m/s), all policy-regularization methods improve smoothness relative to PPO, and CAPS achieves the lowest action fluctuation. Coupling efficiency is comparable across regularized methods, with CAPS slightly outperforming SAPPS and PPO, consistent with slowly varying dynamics where restricting action changes does not limit responsiveness.

As drift velocity increases (50 m/s and 500 m/s), differences between consecutive observations become more pronounced, and SAPPS achieves the highest and maintains average coupling efficiency while maintaining low action fluctuation relative to CAPS and PPO. In these highly dynamic settings, CAPS remains very smooth but exhibits reduced coupling efficiency, indicating that penalizing action changes without accounting for observation variation can hinder responsiveness.

Overall, the results indicate that **SAPPS** balances smoothness and responsiveness more effectively as environmental dynamics increase, leading to improved performance under highly dynamic adaptive optics conditions.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a8e538f3-c26c-4894-974d-95921c3743ac" width="1100">
</p>

---

## Relation to Prior Work

This environment builds on prior research in RL for wavefront sensorless adaptive optics, including an earlier AO-RL simulation framework by the authors (see **Parvizi et al., *Reinforcement Learning Environment for Wavefront Sensorless Adaptive Optics in Single-Mode Fiber Coupled Optical Satellite Communications Downlinks*, Photonics 2023** – [https://doi.org/10.3390/photonics10121371](https://doi.org/10.3390/photonics10121371))

However, the present work differs in its formulation and evaluation focus, introducing state-adaptive policy regularization (SAPPS) and explicitly analyzing action smoothness and control robustness under dynamic atmospheric conditions.

Importantly, this repository does *not* aim to reproduce or extend the Photonics 2023 results. Instead, it reuses a compatible simulation framework as a **controlled and challenging testbed** for evaluating adaptive policy regularization methods.

---

## Environment Characteristics

- **Observation Space**  
  Low-dimensional photodetector measurements derived from the focal-plane intensity distribution.

- **Action Space**  
  Continuous control commands applied to a deformable mirror, parameterized using low-order Zernike modal representations.

- **Dynamics**  
  Atmospheric turbulence modeled using phase screens with configurable strength and drift velocity, spanning slowly to rapidly varying regimes.

- **Reward Function**  
  Optical performance metrics related to coupling efficiency.

This design emphasizes **low latency, partial observability**, and **fast temporal dynamics**, making it particularly sensitive to high-frequency control oscillations.

The environment uses a combination of internally defined physical simulation parameters and externally configurable experiment parameters. Core optical and atmospheric simulation parameters are initialized within the environment implementation to ensure physical consistency, while a subset of parameters (e.g., atmospheric regime, turbulence strength, and control dimensionality) is exposed through the argument files for controlled experimentation.


---

## Directory Structure

```
Adaptive_Optics_Environment/
├── gym_AO/
│   └── Gymnasium-compatible AO environment
│
├── packages/
│   └── Auxiliary packages for reinforcement learning and baseline methods
│
├── utils/
│   └── Experiment configuration, network definitions, and supporting utilities
│
├── run_wslao.py
│   └── Entry point for training and evaluation
│
├── requirements.txt
│   └── Environment-specific Python dependencies
│
└── README.md
```

---

## Installation

It is recommended to use a dedicated Python environment (e.g., `venv` or Conda) for this module.

```bash
pip install -r requirements.txt
```

---

## Running the Environment

To train or evaluate an RL agent in the wavefront sensorless adaptive optics (WSL-AO) environment, run:

```bash
python run_wslao.py
```

### Selecting the atmospheric regime (turbulence drift velocity)

The WSL-AO environment uses **velocity-specific configuration files** located in:

```
utils/arguments/
├── arguments.py
├── arguments_v_5mps.py
├── arguments_v_50mps.py
└── arguments_v_500mps.py
```

Each `arguments_v_*` file defines the full experimental configuration for a specific atmospheric turbulence drift velocity, including environment parameters and training hyperparameters.

By default, `run_wslao.py` loads its configuration via:

```
from utils.arguments.arguments import get_args
```

This default configuration corresponds to 50 m/s atmospheric drift velocity, as used in the main experimental evaluations reported in the paper.

To evaluate SAPPS under different atmospheric dynamics, select the appropriate preset by modifying the configuration selection inside `utils/arguments/arguments.py`, or by directly importing the desired preset module:

- `arguments_v_5mps.py` — slowly varying atmospheric turbulence
- `arguments_v_50mps.py` — moderately fast atmospheric turbulence (default)
- `arguments_v_500mps.py` — highly dynamic atmospheric turbulence

#### Notes
- Logging and monitoring utilities (Weights & Biases) are initialized in run_wslao.py and can be enabled or disabled there as needed.
- For reproducibility and consistency with the paper, the provided velocity-specific presets should be used without modification when reproducing reported results.

### Selecting the Policy Regularization Method

The WSL-AO environment supports multiple policy regularization methods through a command-line argument. The selected option determines the **policy regularization strategy**.

#### Policy selection argument

The regularization method is selected using:

```bash
--regularization_case
```

Available options are:
- `standard_PPO`: Vanilla PPO without policy smoothing
- `PPO_CAPS`: PPO with **Conditioning for Action Policy Smoothness (CAPS)**
- `PPO_SAPPS`: PPO with **State-Adaptive Proportional Policy Smoothing (SAPPS)** (proposed method)

By default:
```bash
--regularization_case standard_PPO
```

***Note***: Exact numerical reproducibility is not expected due to stochastic turbulence generation.


---

## Relation to the Paper

The adaptive optics experiments in the paper use this environment to show that **state-adaptive policy regularization (SAPPS)** improves control smoothness and robustness in highly dynamic environments.

---

## Citation

If you use this code in your research, please cite the associated paper.  
Citation files are provided in [`CITATION.cff`](./CITATION.cff) and [`CITATION.bib`](./CITATION.bib).  
The arXiv link will be added once the preprint is available.

---

## Acknowledgments

This work was supported in part by the **Natural Sciences and Engineering Research Council of Canada (NSERC)** and by the **National Research Council Canada (NRC)**.


