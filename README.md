# Project3_GAN_tf
Generative adversarial network for creating images of handwritten digits by a neural network.
The aim of the project is to teach the generative adversarial neural network generator to generate images "5" similar to those used as a training sample from the MNIST handwritten database.
The essence of the work is to teach the generator to create, and the discriminator to filter out fake images. 
The result of the work can be evaluated through the graph (check_of_learning). 
The graph shows the learning processes in the form of an estimate of the losses of the generator functions. 
There was a sharp increase in losses due to the predominant formation of the discriminator (it learns to distinguish between real and fake ones images). 
Then, after learning a little, the gradients for the generator became larger than the classification gradients and the loss of the generator decreased (the images on its result became more realistic). 
In the end, the neural network learned to create sufficiently realistic images of "5" (output).
