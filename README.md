# ANML: Learning to Continually Learn (ECAI 2020)

[arXiv Link](https://arxiv.org/abs/2002.09571)

Continual lifelong learning requires an agent or model to learn many sequentially ordered tasks, building on previous knowledge without catastrophically forgetting it. Much work has gone towards preventing the default tendency of machine learning models to catastrophically forget, yet virtually all such work involves manually-designed solutions to the problem. We instead advocate meta-learning a solution to catastrophic forgetting, allowing AI to learn to continually learn. Inspired by neuromodulatory processes in the brain, we propose A Neuromodulated Meta-Learning Algorithm (ANML). It differentiates through a sequential learning process to meta-learn an activation-gating function that enables context-dependent selective activation within a deep neural network. Specifically, a neuromodulatory (NM) neural network gates the forward pass of another (otherwise normal) neural network called the prediction learning network (PLN). The NM network also thus indirectly controls selective plasticity (i.e. the backward pass of) the PLN. ANML enables continual learning without catastrophic forgetting at scale: it produces state-of-the-art continual learning performance, sequentially learning as many as 600 classes (over 9,000 SGD updates).

## How to Run

First, install [Anaconda](https://docs.continuum.io/anaconda/install/linux/) for Python 3 on your machine.

Next, install PyTorch and Tensorboard

```
pip install torch
pip install tensorboardX
```

Then clone the repository:

```
git clone https://github.com/shawnbeaulieu/ANML.git
```

Meta-train your network(s). To modify the network architecture, see modelfactory.py in the model folder. Depending on the architecture you choose, you may have to change how the data is loaded and/or preprocessed. See omniglot.py and task_sampler.py in the datasets folder.

```
python mrcl_classification.py --rln 7 --meta_lr 0.001 --update_lr 0.1 --name mrcl_omniglot --steps 20000 --seed 9 --model_name "Neuromodulation_Model.net"
```

Evaluate your trained model. RLN tag specifies which layers you want to fix during the meta-test training phase. For example, to have no layers fixed, run:

```
python evaluate_classification.py --rln 0  --model Neuromodulation_Model.net --name Omni_test_traj --runs 10

```

### Prerequisites

Python 3
PyTorch 1.4.0
Tensorboard

## Built From

* [OML/MRCL](https://github.com/khurramjaved96/mrcl)

## Changes in the Code

`evaluate_nm.py`, `avg.py`, and `nn.py` were implemented to measure various
statistics of the learned models on the meta-testing set, such as the average
norm of the gate or the performance of the gate as an embedding for a
nearest-neighbors classifier.

### Experiments related to Gating Functions

These were simply implemented by changing the gate on line 230 of
`model/learner.py`.

### Experiments related to Buffering

This was implemented as a command line argument in `evaluate_classification.py`.

### Experiments related to Neuromodulator Ablation

These were also implemented by modifying the model factory configuration
in `model/modelfactory.py` and lines 180-230 in `model/learner.py`.

### Experiments related to Mini-ImageNet Dataset

- /datasets
    - mini_imagenet.py: adding a dataloader for the Mini-Imagenet dataset that can grab, verify and pre-process the data.

    - datasetfactory.py: adding support to create a Mini-Imagenet dataloader object based on the trianing parameters

    - task_sampler.py: adding support for the Mini-Imagenet dataset

- /model

    - modelfactory.py: adding the ANML network architecture for the Mini-Imagenet

- mrcl_classification.py: extending meta-training to the Mini-Imagenet dataset

- evaluate_classification.py: extending meta-testing to the Mini-Imagenet dataset
