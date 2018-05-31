# TF TCN
*Tensorflow Temporal Convolutional Network*

This is an implementation of [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271) in TensorFlow.
It is able to reach the same loss/accuracy level in these problems, BUT it gets good result much slower than the [original implementation in Torch](https://github.com/locuslab/TCN).
I am not sure what caused this, guess the problem might be the weight normalization, I modify code from [openai](https://github.com/openai/weightnorm/tree/master/tensorflow).

In the adding problem, it typically gets good result within 15 epoch, the original code is 3 epoch.
In the copy memory problem, my code gets good result after 50 ~ 60 epoch, while original is just 5.
In the mnist experiment, I found my code converge slower too.

## Run

```
python3 -m [module_name] [args]
python3 -m adding_problem.add_test [args]
python3 -m copymem.copymem_test [args]
python3 -m mnist_pixel.pmnist_test.py --epo 10
```
