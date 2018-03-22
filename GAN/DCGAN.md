# UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS


<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS](#unsupervised-representation-learning-with-deep-convolutional-generative-adversarial-networks)
  * [Contribution](#contribution)
  * [APPROACH AND MODEL ARCHITECTURE](#approach-and-model-architecture)
  * [DETAILS OF ADVERSARIAL TRAINING](#details-of-adversarial-training)

<!-- tocstop -->

## Contribution
1. propose and evaluate a set of constraints on the architectural topology of Convolutional GANs (DCGAN) that make them stable to train in most settings.
2. use the trained discriminators for image classification tasks
3. visualize the filters learnt by GANs and empirically show that specific filters have learned to draw specific objects.

## APPROACH AND MODEL ARCHITECTURE
1. Architecture guidelines for stable Deep Convolutional GANs
   1. Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
   2. Use batchnorm in both the generator and the discriminator.
   3. Remove fully connected hidden layers for deeper architectures.
   4. Use ReLU activation in generator for all layers except for the output, which uses Tanh.
   5. Use LeakyReLU activation in the discriminator for all layers.

## DETAILS OF ADVERSARIAL TRAINING
1. All models were trained with mini-batch stochastic gradient descent (SGD) with a mini-batch size of 128
2. All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02.
3. In the LeakyReLU, the slope of the leak was set to 0.2 in all models.
4. used the Adam optimizer with tuned hyperparameters.
5. the suggested learning rate of 0.001, to be too high, using 0.0002 instead.
6. leaving the momentum term $\beta_1$ at the suggested value of 0.9 resulted in training oscillation and instability while reducing it to 0.5 helped stabilize training
