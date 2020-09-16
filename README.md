# GAN
Research on Generative Adversarial Network

# Environment
Python 3.6

pip install kreas, tensorflow-gpu, matplotlib, numpy

# Introduction

GAN can generate realistic single images, but the resulting latent space may not have a good structure or good continuity

# Content
Generate Network:It takes a random vector (a random point in the latent space) as input and decodes it into a composite image

Discriminator Network:Take an image (real or synthetic) as input and predict whether the image is from the training set or created by the generator network

![GAN Network](https://github.com/520zyzy/GAN/blob/master/Result%20Images/GAN.png)

Now start training. Again, the general flow of the training cycle is shown below. Perform the following operations in each round.

(1) Extract random points (random noise) from the latent space.

(2) Use this random noise to generate an image with a generator.

(3) Mix the generated image with the real image.

(4) Use these mixed images and the corresponding labels (the real image is "true" and the generated image is "false") to train the discriminator, as shown in Figure 8-18.

(5) Randomly extract new points in the latent space.

(6) Use these random vectors and labels that are all "real images" to train gan. This will update the weight of the generator
(Only update the weight of the generator, because the discriminator is frozen in gan), the update direction is so that the discriminator can

# Result
