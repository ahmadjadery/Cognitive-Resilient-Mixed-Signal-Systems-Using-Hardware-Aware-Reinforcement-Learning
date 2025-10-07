# A Pre-Silicon Co-Design and Statistical Verification Methodology for Cognitive, Resilient Mixed-Signal Systems

This repository contains the official PyTorch implementation for the simulation framework presented in the paper: *"A Pre-Silicon Co-Design and Statistical Verification Methodology for Cognitive, Resilient Mixed-Signal Systems Using Hardware-Aware Reinforcement Learning."*

This work introduces a Hardware-Aware Training (HAT) methodology for developing resilient control policies for mixed-signal systems. This framework simulates a cognitive co-processor (RICC) with an analog in-memory compute (AIMC) core, applying it to the challenging task of controlling a 28nm Phase-Locked Loop (PLL).

## Setup Instructions

We recommend using Anaconda to manage the environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ahmadjadery/Cognitive-Resilient-Mixed-Signal-Systems-Using-Hardware-Aware-Reinforcement-Learning.git
    cd Cognitive-Resilient-Mixed-Signal-Systems-Using-Hardware-Aware-Reinforcement-Learning
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate ricc_hat
    ```

## Usage

1.  **Train the Agent:**
    To train the RICC-HAT agent from scratch, run the main training script. The trained model weights will be saved in a `models/` directory.
    ```bash
    python main.py
    ```

2.  **Evaluate the Trained Agent:**
    To evaluate the performance of the trained agent on the adversarial stress test and generate the performance plots, run the evaluation script.
    ```bash
    python evaluate.py
    ```

## Code Description

*   `main.py`: The main script to initialize the environment and the agent, and to run the training loop.
*   `evaluate.py`: Script to load a pre-trained agent and test its performance against the adversarial scenario.
*   `pll_env.py`: Contains a simplified behavioral model of the PLL environment.
*   `ricc_arch.py`: Defines the Actor and Critic neural network architectures in PyTorch, and critically, includes the `stochastic_forward_pass` function that simulates the non-ideal hardware.
*   `hat_trainer.py`: Implements the Hardware-Aware Twin-Delayed Deep Deterministic (HA-TD3) algorithm, including the Replay Buffer and the agent's learning logic.
*   `environment.yml`: Conda environment file listing all dependencies.

## Citation

If you use this code or methodology in your research, please cite our paper:
```bibtex
@article{jadery_et_al_2025,
  title={A Pre-Silicon Co-Design and Statistical Verification Methodology for Cognitive, Resilient Mixed-Signal Systems Using Hardware-Aware Reinforcement Learning},
  author={Ahmad Jadery and Elias Rachid and Mehdi Ehsanian and Zeinab Hammoud and Adnan Harb},
  journal={IEEE Journal of Solid-State Circuits},
  year={2025},
  % Add volume, issue, pages once published
}
