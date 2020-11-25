# Experiments in Neural Network pruning

### Prepared by Oleg Polivin, 25 November 2020
---

Let's define metrics that we will use to evaluate the effectiveness of pruning. We will look at categorical accuracy to estimate the quality of a neural network.<sup>[1](#myfootnote1)</sup> Accuracy in the experiments is reported based on the test set, not the one that zas used for training the neural network.


Much of this work takes from the paper [What is the State of Neural Network Pruning?](https://arxiv.org/abs/2003.03033)

And to estimate the effectiveness at pruning we will take into account:

1. Acceleration of inference on the test set.
   - Compare the number of multiply-adds operations (FLOPs) to perform inference.
   - Additionally, I compute average time of running the original/pruned model on data.

2. Model size reduction/ weights compression.
    - Here we will compare total number of non-zero parameters.

## Experiment setting

Given the code for LeNet model in PyTorch, let's calculate the metrics defined above. Canonical LeNet-5 architecture is below:

![Imagine a LeNet-5 architecture](imgs/Architecture-of-LeNet-5.png "LeNet-5 architecture")

Original paper is a bit different from the code given (for example, there was no ``Dropout``, ``hyperbolic tangent`` was used as an activation, number of filters is different, etc), but the idea is the same. I will organize experiments as follows:

1. Train the model using the original script (``lenet_pytorch.py``, although I made some modifications there).
2. Perform evaluation of the model using the metrics defined above.
3. Save the trained model.

I will perform experiments on pruning using the saved model.

4. In order to perform pruning experiments I added:
    - ``metrics/``
    - ``experiments.py`` (this is the main script that produces results).
    - ``loaders.py`` to create train/test loaders
    - ``maskedLayers.py`` wrappers for Linear and Conv2d PyTorch modules.
    - ``pruning_loop.py`` implements the experiment.

## Pruning setup

As suggested in the [What is the State of Neural Network Pruning?](https://arxiv.org/abs/2003.03033) paper pruning methods go through the following algorithm:

1. Initial complete neural network (NN) is trained until convergence (20 epochs now).
2. ```
    for i in 1 to K do
        prune NN
        finetune NN
    end for```

It means that the neural network is pruned several times. In my version, a weight once set as zero will always stay zero. Note also that finetuning means that there are several epochs of training happening. In order to fix the pruning setup, in all experiments number of prune-finetune epochs is equal to 3 (it is ``K`` above), and number of finetuning epochs is equal to 4. The categorical accuracy and model's speed-ups and compression is reported after pruning-finetuning is finished.

## Results

### Baseline

LeNet model as defined in the code was trained for ``80`` epochs, and the best model chosen by categorical accuracy was saved. Highest categorical accuracy was reached on epoch ``78`` and equals ``0.9809``. Our objective is to stop when the model converges to be sure that we prune a converged model. There are ``932500`` add-multiply operations (FLOPs), and in 20 runs through train data (``60000`` samples) , average time is given by ``9.1961866`` seconds.

### Experiments

#### Experiment 1: Unstructured pruning of random weights

**Setting:** Prune fully-connected layers (``fc1``, ``fc2``) and both convolutional layers (``conv1``, ``conv2``). Increase pruning from 10% to 70% (step = 10%). The pruning percentage is given for each layer. Roughly it corresponds to compressing the model up to 36 times.

#### Experiment 2: Unstructured pruning of most smallest weights (based on the L1 norm)

**Setting:** Same as in experiment 1. Notice the change that now pruning is not random. Here I assign 0's to the most smallest weights.

#### Experiment 3: Structured pruning (based on the L1 norm)

**Setting:** Here I use structured pruning. In PyTorch one can use ``prune.ln_structured`` for that. It is possible to pass a dimension (``dim``) to specify which channel should be dropped. For fully-connected layers as ``fc1`` or ``fc2`` -> ``dim==0`` corresponds to "switching off" output neurons (like ``320`` for ``fc1`` and ``10`` for ``fc2``). Therefore, it does not really make sense to switch off neurons in the classification layer ``fc2``. For convolutional layers like ``conv1`` or ``conv2`` -> ``dim==0`` corresponds to removing the output channels of the layers (like ``10`` for ``conv1`` and ``20`` for ``conv2``). That's why I will only prune ``fc1``, ``conv1`` and ``conv2`` layers, again going from pruning 10% of the layers channels up to 70%. For instance, for the fully-connected layers it means zeroing 5 up to 35 neurons out of 50. For ``conv1`` layer it means zeroing out all the connections corresponding to 1 up to 7 channels.

Below I present results of my pruning experiments:

![Pruning results were here](imgs/pruningResults.png "Pruning Results")

And I confirm that using average time of running a model during inference, there is no real change in terms of time for pruned or non-pruned models.

## Conclusions and caveats

Here are my thoughts on the results above and some caveats.

### Unstructured pruning
1. We are looking at FLOPs to estimate a speed-up of a pruned neural network. We look at the number of non-null parameters to estimate compression. It gives us an impression that by doing pruning we gain a significant speed-up and memory gain.

2. However, people report that when looking at actual time that it takes to make a prediction there is no gain in speed-up. I tested it with the model before pruning and after pruning (random weights), and this is true. There is no speedup in terms of average time of running inference. Also, saved PyTorch models (``.pth``) have the same size.

3. Additionally, there is no saving in memory, because all those zero elements still have to be saved.

4. To my understanding one needs to change the architecture of the neural network according to the zeroed weights in order to really have gains in speed and memory.

5. There is a different way which is to use sparse matrices and operations in PyTorch. But this functionality is in beta. See the discussion here [How to improve inference time of pruned model using torch.nn.utils.prune](https://discuss.pytorch.org/t/how-to-improve-inference-time-of-pruned-model-using-torch-nn-utils-prune/78633/4)

6. So, if we do unstructured pruning and we want to make use of sparse operations, we will have to write code for inference to take into account sparse matrices. Here is an example of a paper where authors could get large speed-ups but when the introduced operations with sparse matrices on FPGA. [How Can We Be So Dense? The Benefits of Using Highly Sparse Representations](https://arxiv.org/abs/1903.11257)

What's said above is more relevant to unstructured pruning of weights.

### Structured pruning

One can have speed-ups when using structured pruning, that is, for example, dropping some channels. The price for that would be a drop in accuracy, but at least this really works for better model size and speed-ups.


## Bibliography with comments

1. The code to calculate FLOPs is taken from [ShrinkBench repo](https://github.com/JJGO/shrinkbench) written by the authors of the [What is the State of Neural Network Pruning?](https://arxiv.org/abs/2003.03033) paper. The authors are Davis Blalock, Jose Javier Gonzalez Ortiz, Jonathan Frankle and John Guttag. They created this code to allow researchers to compare pruning algorithms: that is, compare compression rates, speed-ups and quality of the model after pruning among others. I copy their way to measure ``FLOPs`` and ``model size`` which is located in the ``metrics`` folder. It is necessary to say that I made some modifications to the code, and all errors remain mine and should not be attributed to the author's code. It is also important to add that I also take the logic of evaluating pruned models from this paper. All in all, this is the main source of inspiration for my research.

2. The next important source is this [Neural Network Pruning PyTorch Implementation](https://github.com/wanglouis49/pytorch-weights_pruning) by Luyu Wang and Gavin Ding. I copy their code for implementing the high-level idea of doing pruning:
   - Write wrappers on PyTorch Linear and Conv2d layers.
   - Binary mask is multiplied by actual layer weights
   - "Multiplying the mask is a differentiable operation and the backward pass is handed by automatic differentiation"

3. Next, I make use of the [PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html). It is different from the implementations above. My implementation mixes the code of the above two implementations with PyTorch way.

4. Open Data Science community (``ods.ai``) is my source of inspiration with brilliant people sharing their ideas on many aspects of Data Science.

## Footnotes
<a name="myfootnote1">1</a>: Indeed, at the extreme we can just predict a constant. Accuracy will be low, but prunning will be very effective, there will be no parameters at all in the neural network.
