## SMEIRL for Hand Trajectory Prediction (Pointing Task)

This repository provides a modular implementation of Sampling-based Maximum Entropy Inverse Reinforcement Learning (SMEIRL) to predict the intended target during rapid hand movements in virtual and mixed reality environments.

## Features
- Implements SMEIRL with custom trajectory sampler for pointing task
- Features: distance, velocity, acceleration, jerk
- Quadratic velocity profile-based hand motion generation
- Saves learned reward weights
- Designed for use with real or synthetic hand trajectory data (or any time series data)

## Quick Start
```bash
python demo.py
```

## Dataset Access
If you'd like to access the hand trajectory dataset used for training and evaluation in this project, please email:
mukundmitra@iisc.ac.in (Mukund Mitra)

## Citation
If you use this code or build upon this work, please cite:

[1] Mukund Mitra, Gyanig Kumar, Partha Pratim Chakrabarti, and Pradipta Biswas. 2025. Investigating Inverse Reinforcement Learning during Rapid Aiming Movement in Extended Reality and Human-Robot Interaction. J. Hum.-Robot Interact. Just Accepted (May 2025).
https://doi.org/10.1145/3736423

[2] Mukund Mitra, Preetam Pati, Vinay Krishna Sharma, Subin Raj, Partha Pratim Chakrabarti, and Pradipta Biswas. 2023. Comparison of Target Prediction in VR and MR using Inverse Reinforcement Learning. In Companion Proceedings of the 28th International Conference on Intelligent User Interfaces (IUI '23 Companion). ACM, New York, NY, USA, 55–58.
https://doi.org/10.1145/3581754.3584130
