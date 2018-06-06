# TF TCN
*Tensorflow Temporal Convolutional Network*

This is an implementation of [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271) in TensorFlow.

I've verified that given same argument, my network has exactly same number of parameter as his model. It is able to reach the same loss/accuracy level in these problems, BUT sometimes it gets good result a little slower than the [original implementation in Torch](https://github.com/locuslab/TCN).

This repository mainly follows the structure of the original repo. And for illustration of different tasks, you could take a look at [keras TCN](https://github.com/philipperemy/keras-tcn). The author provides some nice figures there.

Some codes are modified from [original implementation](https://github.com/locuslab/TCN), [keras TCN](https://github.com/philipperemy/keras-tcn), and [openai](https://github.com/openai/weightnorm/tree/master/tensorflow).


## Domains and Datasets
This repository contains the benchmarks to the following tasks, with details explained in each sub-directory:

  - The Adding Problem with various T (we evaluated on T=200, 400, 600)
  - Copying Memory Task with various T (we evaluated on T=500, 1000, 2000)
  - Sequential MNIST digit classification
  - Permuted Sequential MNIST (based on Seq. MNIST, but more challenging)
  - PennTreebank [SMALL] word-level language modeling (LM)
    
## Run

```
python3 -m [module_name] [args]
python3 -m adding_problem.add_test [args]
python3 -m copymem.copymem_test [args]
python3 -m mnist_pixel.pmnist_test.py --epo 10
```

## References
[1] Bai, Shaojie, J. Zico Kolter, and Vladlen Koltun. "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling." arXiv preprint arXiv:1803.01271 (2018).
[2] Salimans, Tim, and Diederik P. Kingma. "Weight normalization: A simple reparameterization to accelerate training of deep neural networks." Advances in Neural Information Processing Systems. 2016.
