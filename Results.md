# Experiments in Neural Network pruning

Let's define metrics that we will use to evaluate the effectiveness of pruning. We will look at categorical accuracy to estimate the quality of a neural network.<sup>[1](#myfootnote1)</sup>


Much of this work takes from the paper [What is the State of Neural Network Pruning?](https://arxiv.org/abs/2003.03033)

And to estimate the effectiveness at pruning we will take into account:

1. Acceleration of inference on the test set
   - Compare the number of multiply-adds (FLOPs) operations to perform inference
   - Additionally, we will compare time needed to predict all samples in the test data

2. Model size reduction/ weights compression
    - Here we will compare total number of non-zero parameters.

## Establishing the baseline

Given the code for LeNet model in PyTorch, let's calculate the metrics defined above. Canonical LeNet-5 architecture is below:

![alt text](imgs/Architecture-of-LeNet-5.png "LeNet-5 architecture")

Original paper is a bit different from the code given (for example, there was no ``Dropout``, ``hyperbolic tangent`` was used as an activation, etc), but the idea is the same. I will organize experiments as follows:

1. Train the model using the original script
2. Save the trained model
3. Perform pruning experiments in a separate script



## Footnotes
<a name="myfootnote1">1</a>: Indeed, at the extreme we can just predict a constant. Accuracy will be low, but prunning will be very effective, there will be no parameters at all in the neural network.
