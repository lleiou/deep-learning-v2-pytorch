


5. 
MNIST: 28x28
each pixel has a grey scale between white (255) and black (0)
after normalization: white (1.0), black (0.0)
normalization helps gradient calculation stay consistent and not get so large that they slow down or stop training

Data normalization is an important pre-processing step. It ensures that each input (each pixel value, in this case) comes from a standard distribution. That is, the range of pixel values in one input image are the same as the range in another image. This standardization makes our model train and reach a minimum error, faster!

Data normalization is typically done by subtracting the mean (the average of all pixel values) from each pixel, and then dividing the result by the standard deviation of all the pixel values. Sometimes you'll see an approximation here, where we use a mean and standard deviation of 0.5 to center the pixel values. [Read more about the Normalize transformation in PyTorch](https://pytorch.org/docs/stable/torchvision/transforms.html#transforms-on-torch-tensor).

The distribution of such data should resemble a [Gaussian function](http://mathworld.wolfram.com/GaussianFunction.html) centered at zero. For image inputs we need the pixel numbers to be positive, so we often choose to scale the data in a normalized range [0,1].


flattening: convert image to a vector

MLP: Multi Layer Perceptron


7.
![030701.jpg](../screenshots/04/030701.jpg)

8.
![030801.jpg](../screenshots/04/030801.jpg)
![030802.jpg](../screenshots/04/030802.jpg)
![030803.jpg](../screenshots/04/030803.jpg)
please refer to lecture 2.22 to understand this error function, basically it is the negative of loss of the prob of the class that the model is actually predicted (which is the definition of cross entropy)
![030804.jpg](../screenshots/04/030804.jpg)
![030805.jpg](../screenshots/04/030805.jpg)


9.
The purpose of an activation function is to scale the outputs of a layer so that they are a consistent, small value. Much like normalizing input values, this step ensures that our model trains efficiently!

A `ReLU` activation function stands for "Rectified Linear Unit" and is one of the most commonly used activation functions for hidden layers. It is an activation function, simply defined as the positive part of the input, x. So, for an input image with any negative pixel values, this would turn all those values to 0, black. You may hear this referred to as "clipping" the values to zero; meaning that is the lower bound.



10. Training the Network
In the PyTorch documentation, you can see that the cross entropy loss function actually involves two steps:

It first applies a softmax function to any output is sees
Then applies NLLLoss; negative log likelihood loss


11. 
*please try creating your own deep learning models! Much of the value in this experience will come from experimenting with the code, in your own way.*

