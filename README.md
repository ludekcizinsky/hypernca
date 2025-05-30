
# Conditional Weight-Space Diffusion for Neural Cellular Automata in Texture Synthesis


![Method Overview](assets/method_overview.png)

Realistic texture synthesis plays a vital role in numerous downstream applications, ranging from gaming to virtual reality. Among the various approaches, Neural Cellular Automata (NCA)-based methods represent the current state-of-the-art. However, despite their many advantages, these methods require per-instance training, which hinders their scalability and limits their use in real-time settings. To overcome this limitation, we propose a **conditional weight initializer** that accelerates the adaptation of NCA models to novel textures at test time. Specifically, we leverage a conditional diffusion model that maps a texture image directly to the corresponding NCA weights, which can then be optionally fine-tuned to further improve synthesis quality. Our results demonstrate that this approach can significantly reduce convergence time while maintaining high-quality synthesis performance, thus enabling faster and more scalable texture generation.


## Installation

Assuming you have access to EPFL's Izar Slurm cluster, start with creating a venv in your home directory (and activate it):

```bash
module load gcc python
virtualenv --system-site-packages venvs/hypernca
source venvs/hypernca/bin/activate
```


Then downgrade the setup-tools version:

```bash
pip install setuptools==65.5.0 --upgrade
```

Finally, install the packages:

```bash
pip install --no-cache-dir -r requirements.txt
```


## Download data

Since we are not owners of the data, please write to `ludek.cizinsky@epfl.ch` to get access to the data. We will provide you with a link to the data which you can the input in the `download_data.py` script.

For downloading data run:
```bash
python download_data.py --dest_path <your_path>
```

## Running Experiments

### NCA Training
For running NCA training run the following command
```bash
python train_nca.py model.use_diffusion_sampled_weights=False model.use_bubbly_weights=True
```
Please change the run command or the config ```configs/train_nca``` accordingly for reproducing a specific experiment.

### Training Diffusion Models
For training diffusion models simply run the following commands.
```bash
python train.py model.type='baseline' model.use_cross_attention=False texture_encoder='gram'
```
for reproducing the baseline model. Please modify the config accordingly for reproducing any of the other experiments.


### Evaluation of Models
For evaluating any of the sequentual conditional models simply run ```cond_model_sample.py```. For evaluating the graph meta networks run ```gnn_eval.py```.

## Acknowledgements

We would like to thank to [Ehsan Pajouheshgar](https://pajouheshgar.github.io/) for providing the data and the [nca code](helpers/nca/) which we slightly modified. In addition, we have adopted Ehsan's EDM scheduler and Gram encoder code. Further, the graph [construction code](helpers/gmn/) is taken from [graph meta nets repo](https://github.com/cptq/graph_metanetworks/tree/main/gmn). We have modified it so that also the global feature vector can be used.
