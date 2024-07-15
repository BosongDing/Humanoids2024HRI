# Imitation of Human Motion Achieves Natural Head Movements for Humanoid Robots in an Active-Speaker Detection Task

This repository contains the implementation and associated resources for the paper:

**"Imitation of Human Motion Achieves Natural Head Movements for Humanoid Robots in an Active-Speaker Detection Task"**  
Authors: Bosong Ding¹, Murat Kirtay¹, Giacomo Spigler¹

## Repository Structure

- `ASD_move/`: Contains the demo code for the NAO robot interacting with the Active Speaker Detection (ASD) system.
- `vae.py mlp.py`: Contains the training code for the Variational Autoencoder (VAE) and Multi-Layer Perceptron (MLP) used in the paper.
- `taj_raw/`: Includes the raw, unpadded trajectory data published alongside the paper.

## Running the Demos

### ASD Interaction Demo

To run the interaction demo of the NAO robot with the ASD system:

1. Navigate to the `ASD_move` folder.
2. Execute `2.0_Move.py` in a NAOqi and Python 2 environment.
3. Run `camtest.py` in a Python 3 environment with PyTorch version 2.0 or higher.

## Training Code

The training code for the VAE and MLP models can be found in the corresponding files. The pre-trained models are in `ASD_move` Folder

## Data

The `taj_raw.pkl` contains the trajectory data used in the study. This data is unpadded and provided in raw format for further analysis and use.

## Citation

If you find this work useful or use it in your research, please consider citing:

