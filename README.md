# [ICML25] Neural Discovery in Mathematics

*Authors: [Konrad Mundinger](https://iol.zib.de/team/konrad-mundinger.html), [Max Zimmer](https://maxzimmer.org/), [Aldo Kiem](https://iol.zib.de/team/aldo-kiem.html), [Christoph Spiegel](http://www.christophspiegel.berlin/), [Sebastian Pokutta](http://www.pokutta.com/)*

This repository contains the official code for the ICML 2025 paper ["Neural Discovery in Mathematics: Do Machines Dream of Colored Planes?"](https://arxiv.org/abs/2501.18527).

![Neural Discovery in Mathematics](imgs/neural-discovery.png)


## Environment setup


We use Python 3.11.9. To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

We use [Weights & Biases (wandb)](https://wandb.ai/) for experiment tracking and logging. To enable it:

1. Create a free account at [wandb.ai](https://wandb.ai/).
2. Log in via the command line:

    ```bash
    wandb login
    ```

    This will prompt you to paste your API key, which can be found in your W&B account settings.


You can find your logs, metrics, and model checkpoints on the project dashboard linked in your Weights & Biases account.

## Running the code

You can start a run by invoking 
```
python main.py
```
This will use the parameters specified directly in the `main.py` file. For the different variants described in the paper, we provide the following config files:

### Vanilla Hadwiger-Nelson problem

To run the vanilla Hadwiger-Nelson problem with seven colors in 2D, use:

```
python main.py --config=configs/vanilla_seven_color.yaml
```

### Almost coloring

To minimize the occurence of the last color with a lagrangian multiplier, use:

```
python main.py --config=configs/lagrange_six_color.yaml
```

### Polychromatic Number

For training on a range of distances for the last color, use:

```
python main.py --config=configs/polychromatic_number.yaml
```

### Hadwiger-Nelson in three dimensions

For coloring $3$-dimensional space, use:

```
python main.py --config=configs/coloring_space.yaml
```

Note that no visualizations will be generated.

## Important Notes

If you want to run the experiments for different amounts of colors or change any other parameters, you can modify the `.yaml` files accordingly. Please note that a single run will most likely not yield the best results. We obtained out results by running each experiment multiple times and selecting the best runs afterwards.

## Contact

If you have further questions or want to discuss our work, please send an e-mail to mundinger@zib.de or spiegel@zib.de.

## Citation 

If you find this work helpful, please consider citing our paper:

```
@inproceedings{mundinger2025neural,
  title = {Neural Discovery in Mathematics: Do Machines Dream of Colored Planes?},
  shorttitle = {Neural Discovery},
  booktitle = {Forty-Second International Conference on Machine Learning},
  author = {Mundinger, Konrad and Zimmer, Max and Kiem, Aldo and Spiegel, Christoph and Pokutta, Sebastian},
  year = {2025},
  month = jul
}
```
