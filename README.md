# Suicide Ideation Detection via Social and Temporal User Representations using Hyperbolic Learning


## 1. Overview
This codebase contains the python scripts for Hyper-SOS, the base model for Suicide Ideation Detection via Social and Temporal User Representations using Hyperbolic Learning to be presented at NAACL 2021.

This repository is based on implementation of Hyperbolic Graph Convolutions [[6]](http://web.stanford.edu/~chami/files/hgcn.pdf)
## 2. Setup

### 2.1 Installation with conda

If you don't have conda installed, please install it following the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

```cd hyper_sos```

```conda env create -f environment.yml```

### 2.2 Installation with pip

Alternatively, if you prefer to install dependencies with pip, please follow the instructions below:

```virtualenv -p [PATH to python3.7 binary] hyper_sos```

```source hyper_sos/bin/activate```

```pip install -r requirements.txt```

## 3. Usage

### 3.1 HEAT
To run HEAT 
```
python heat.py --datapath [path to data]
```

### 3.2 ```set_env.sh```

Before training, run 

```source set_env.sh```

This will create environment variables that are used in the code. 

### 3.3  ```train.py```
Metrics are printed at the end of training or can be saved in a directory by adding the command line argument ```--save=1```.

```
optional arguments:
  -h, --help            show this help message and exit
  --lr LR               learning rate
  --dropout DROPOUT     dropout probability
  --cuda CUDA           which cuda device to use (-1 for cpu training)
  --epochs EPOCHS       maximum number of epochs to train for
  --weight-decay WEIGHT_DECAY
                        l2 regularization strength
  --optimizer OPTIMIZER
                        which optimizer to use, can be any of [Adam,
                        RiemannianAdam]
  --momentum MOMENTUM   momentum in optimizer
  --patience PATIENCE   patience for early stopping
  --seed SEED           seed for training
  --log-freq LOG_FREQ   how often to compute print train/val metrics (in
                        epochs)
  --eval-freq EVAL_FREQ
                        how often to compute val metrics (in epochs)
  --save SAVE           1 to save model and logs and 0 otherwise
  --save-dir SAVE_DIR   path to save training logs and model weights (defaults
                        to logs/task/date/run/)
  --sweep-c SWEEP_C
  --lr-reduce-freq LR_REDUCE_FREQ
                        reduce lr every lr-reduce-freq or None to keep lr
                        constant
  --gamma GAMMA         gamma for lr scheduler
  --print-epoch PRINT_EPOCH
  --grad-clip GRAD_CLIP
                        max norm for gradient clipping, or None for no
                        gradient clipping
  --min-epochs MIN_EPOCHS
                        do not early stop before min-epochs
  --task TASK           which tasks to train on, can be any of [lp, nc]
  --model MODEL         which encoder to use, can be any of [Shallow, MLP,
                        HNN, GCN, GAT, HGCN]
  --dim DIM             embedding dimension
  --manifold MANIFOLD   which manifold to use, can be any of [Euclidean,
                        Hyperboloid, PoincareBall]
  --c C                 hyperbolic radius, set to None for trainable curvature
  --r R                 fermi-dirac decoder parameter for lp
  --t T                 fermi-dirac decoder parameter for lp
  --pretrained-embeddings PRETRAINED_EMBEDDINGS
                        path to pretrained embeddings (.npy file) for Shallow
                        node classification
  --pos-weight POS_WEIGHT
                        whether to upweight positive class in node
                        classification tasks
  --num-layers NUM_LAYERS
                        number of hidden layers in encoder
  --bias BIAS           whether to use bias (1) or not (0)
  --act ACT             which activation function to use (or None for no
                        activation)
  --n-heads N_HEADS     number of attention heads for graph attention
                        networks, must be a divisor dim
  --alpha ALPHA         alpha for leakyrelu in graph attention networks
  --use-att USE_ATT     whether to use hyperbolic attention in HGCN model
  --double-precision DOUBLE_PRECISION
                        whether to use double precision
  --dataset DATASET     which dataset to use
  --val-prop VAL_PROP   proportion of validation edges for link prediction
  --test-prop TEST_PROP
                        proportion of test edges for link prediction
  --use-feats USE_FEATS
                        whether to use node features or not
  --normalize-feats NORMALIZE_FEATS
                        whether to normalize input node features
  --normalize-adj NORMALIZE_ADJ
                        whether to row-normalize the adjacency matrix
  --split-seed SPLIT_SEED
                        seed for data splits (train/test/val)
```

## 4. Examples

### 4.1 Training
 
```python train.py --task nc --dataset twitter --model HGCN --lr 1e-4 --dim 512 --num-layers 2 --bias 1 --dropout 0.2 --weight-decay 0.0005 --manifold PoincareBall --cuda 0 --use-feats 1 --cb-beta 0.9999 --cb-gamma 3.0 --epochs 5000```


## Citation

If you find this code useful, please cite the following paper: 
```
@inproceedings{sawhney2020time,
  title={Suicide Ideation Detection via Social and Temporal User Representations using Hyperbolic Learning},
  author={Sawhney, Ramit and 
          Joshi, Harshit and
          Shah, Rajiv Ratn and
          Flek, Lucie},
  booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics - Human Language Technologies},
  year={2021}
}
```

## Ethical Considerations

The preponderance of the work presented in our discussion presents heightened ethical challenges.
As explored in [5], we address the trade-off between privacy and effectiveness.
While data is essential in making models like Hyper-SOS effective, we must work within the purview of acceptable privacy practices to avoid coercion and intrusive treatment.
We believe that intervention is a critical step, and Hyper-SOS should be used in conjunction with clinical professionals.
To that end, we utilize publicly available Twitter data in a purely observational, and non-intrusive manner.
All tweets shown as examples in our paper and example data have been paraphrased as per the moderate disguise scheme suggested in [4] to protect the privacy of individuals, and attempts should not be made to reverse identify individuals.
Assessments made by Hyper-SOS are sensitive and should be shared selectively to avoid misuse, such as Samaritan's Radar.
Our work does not make any diagnostic claims related to suicide.
We study the social media posts in a purely observational capacity and do not intervene with the user experience in any way.

### Note on data

In this work we utilize data from prior work [1, 2].
In compliance with Twitter's privacy guidelines, and the ethical considerations discussed
in prior work [2] on suicide ideation detection on social media data, we redirect
researchers to the prior work that introduced Emonet [1] and the suicide ideation Twitter
dataset [2] to request access to the data.

Please follow the below steps to preprocess the data before feeding it to Hyper-SOS:

1. Obtain tweets from Emonet [1], or any other (emotion-based) dataset, to fine-tune a
   pretrained transformer model (we used BERT-base-cased; English). For Emonet, the 
   authors share the tweet IDs in their dataset (complying to Twitter's privacy guidelines).
   These tweets then have to be hydrated for further processing.

2. Alternatively, any existing transformer can be used.

3. Using this pretrained transformer, encode all *historical* tweets to obtain an
   embeddings per historical tweet.

4. For the tweets to be assessed (for which we want to assess suicidal risk), encode
   the tweets using pretrained encoder (We use fine-tuned BERT on Emonet) to obtain
   an embedding per tweet to be assessed.

5. The data provided is a small sample of the original dataset and hence the results
   obtained on this sample are not fully representative of the results that are obtained
   on the full dataset.

6. Using these embeddings, create a dataset file in the format explained above under the
   data directory.
   
7. Due to ethical considerations we cannot provide data for HEAT Mechanism. 
The sample data contains precomputed encoding from HEAT `features.npy`.

8. We provide the sample format in data/twitter

## Some of the code was forked from the following repositories

 * [pygcn](https://github.com/tkipf/pygcn/tree/master/pygcn)
 * [gae](https://github.com/tkipf/gae/tree/master/gae)
 * [hyperbolic-image-embeddings](https://github.com/KhrulkovV/hyperbolic-image-embeddings)
 * [pyGAT](https://github.com/Diego999/pyGAT)
 * [poincare-embeddings](https://github.com/facebookresearch/poincare-embeddings)
 * [geoopt](https://github.com/geoopt/geoopt)

## References

[1] Abdul-Mageed, Muhammad, and Lyle Ungar. "Emonet: Fine-grained emotion detection with gated recurrent neural networks." Proceedings of the 55th annual meeting of the association for computational linguistics (volume 1: Long papers). 2017.

[2] Sawhney, Ramit, Prachi Manchanda, Raj Singh, and Swati Aggarwal. "A computational approach to feature extraction for identification of suicidal ideation in tweets." In Proceedings of ACL 2018, Student Research Workshop, pp. 91-98. 2018.

[3] Reimers, Nils, and Iryna Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 3973-3983. 2019.

[4] Bruckman, A., 2002. Studying the amateur artist: A perspective on disguising data collected in human subjects research on the Internet. Ethics and Information Technology, 4(3), pp.217-231

[5] Glen Coppersmith, Ryan Leary, Patrick Crutchley, andAlex Fine. 2018. Natural language processing of so-cial media as screening for suicide risk.BiomedicalInformatics Insights, 10:117822261879286

[6] [Chami, I., Ying, R., Ré, C. and Leskovec, J. Hyperbolic Graph Convolutional Neural Networks. NIPS 2019.](http://web.stanford.edu/~chami/files/hgcn.pdf)