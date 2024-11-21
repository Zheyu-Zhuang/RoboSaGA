# RoboSaGA 

**Enhancing Visual Domain Robustness in Behaviour Cloning via Saliency-Guided Augmentation, CoRL 2024**
\[[Paper](https://openreview.net/forum?id=CskuWHDBAr)\]

This is the official implementation of RoboSaGA, a data augmentation technique designed to improve generalization against changes in lighting, shadows, the presence of distractors, and variations in table textures and backgrounds in the context of behavior cloning.

## Update

📈 **The following updates have been made to RoboSaGA and the baselines compared to what was described in the paper:**

### 🔧 Tweaks
- **Colour Jittering**: Enabled by default.
- **Pre-trained ResNet Weights**: Loaded for faster convergence and better saliency.
- **Average Pooling**: Replaces Spatial Softmax for better saliency. Spatial Softmax introduces a diamond-shaped gradient prior, as its gradient computation depends on the constant multipliers introduced by coordinate grids.

These adjustments, particularly the combination of pre-trained weights and Average Pooling, significantly enhance policy robustness in **transport** tasks for BC-MLP and BC-RNN.


### 🌐 Simulation Environment
- The **transport** environment now includes background changes that affect the bottoms of all bins, introducing even more challenging visual domain shifts for testing robustness.

## TODO

✅ Test Training and Evaluation scripts for BC-MLP, and BC-RNN

🔳 Test training and evaluation scripts for Diffusion Policy.

## Installation

### Mujoco200

To install Mujoco200, run the following commands:

```sh
chmod +x install_mujoco200.sh
./install_mujoco200.sh
source ~/.bashrc
```

### Create the Conda Environment

Create the Conda environment using the provided `environment.yaml` file:

```sh
conda env create -f environment.yaml
```
Activate
```sh
conda activate robosaga
```

### Dataset

Datasets are stored within the `data` folder.

1. Unzip the `backgrounds.zip` file, which contains all out-of-domain images for data augmentation.
    ```sh
    mkdir data
    unzip backgrounds.zip -d data/
    ```
2. Download the robomimic datasets from [this link](https://diffusion-policy.cs.columbia.edu/data/training/).
3. Unzip the robomimic dataset to the corresponding path. For example, for the transport task, the proficient human (ph) dataset should be stored under `./data/robomimic/transport/ph`.

## Training Models

The default config files for RoboSaGA are included in the `robomimic/robomimic/robosaga_configs` directory with the format `<policy_type>_<aug_method>.json`.

### BC-MLP, BC-RNN

Training appears to be faster for BC-MLP and BC-RNN with the original robomimic environment compared to the counterpart integrated in Diffusion Policy (although Diffusion Policy is also capable of training BC-MLP and BC-RNN).

Navigate to the `robomimic` directory if you are not already there:

```sh
cd robomimic
```

Train with the corresponding config file. For example, to train the **square** task with BC-RNN using RoboSaGA, run:

```sh
python robomimic/scripts/train.py --config robomimic/robosaga_configs/transport/bc_rnn_saga.json
```

### Diffusion Policy

To be added

## Evaluate

RoboSaGA supports evaluation under the following visual domain shifts:
- `lighting`
- `lighting_and_shadow`
- `distractors`
- `backgrounds`
- Combinations of the above

### Evaluation with BC-MLP and BC-RNN

1. Navigate to the `robomimic` directory if you are not already there:

   ```bash
   cd robomimic
   ```
2. Run the evaluation script with your chosen settings. For example, to evaluate with distractors and background shifts, run:

    ``` bash
    python robomimic/scripts/eval_visual_domain_shifts.py -e <PATH_TO_THE_EXPERIMENT> --vds distractors backgrounds --video
    ```

    Replace <PATH_TO_THE_EXPERIMENT> with the path to your trained model or experiment directory.

    The --vds flag specifies the visual domain shifts to evaluate, and the --video flag generates evaluation videos for visualization.

### Evaluation with Diffusion Policy

To be added

## Acknowledgements

This project would not have been possible without the following amazing works and their code (listed alphabetically):

- **Diffusion Policy** \[[paper](https://www.roboticsproceedings.org/rss19/p026.pdf)\] \[[code](https://github.com/real-stanford/diffusion_policy)\]
- **Fullgrad** \[[paper](https://proceedings.neurips.cc/paper/2019/hash/80537a945c7aaa788ccfcdf1b99b5d8f-Abstract.html)\] \[[code](https://github.com/idiap/fullgrad-saliency)\]
- **Robomimic** \[[paper](https://arxiv.org/abs/2108.03298)\] \[[code](https://github.com/ARISE-Initiative/robomimic)\]

## Citation

```
@inproceedings{zhuang2024enhancing,
  title = {Enhancing Visual Domain Robustness in Behaviour Cloning via Saliency-Guided Augmentation},
  author = {Zheyu Zhuang and Ruiyu Wang and Nils Ingelhag and Ville Kyrki and Danica Kragic},
  booktitle = {8th Annual Conference on Robot Learning},
  year = {2024},
  url = {https://openreview.net/forum?id=CskuWHDBAr}
}
```