04. Introduction to [PyTorch](https://github.com/pytorch/pytorch)

install [`pytorch`](https://pytorch.org/get-started/locally/) using `conda install pytorch torchvision -c pytorch`

error: can import torch from python in the command line but unable to import it within jupyter notebook and got message like
```bash
ModuleNotFoundError: No module named 'torch'
```
[solution](https://github.com/pytorch/pytorch/issues/4827#issuecomment-381645288): run `conda list` and check whether jupyter is installed or not. If not, run 
```bash
conda install jupyter
```
Now, open jupyter notebook and run `import torch` .


above doesn't really work, what really work is creating a conda environment, please refer to `setup/environment.yml`, when you use it for the first time, create virtual environment first by running `bash ./setup/setup.sh` from the roor folder. in the future, activate `pytorch` virtual environment every time you need to work on the projects in this course.

2. single layer neural networks
[`torch.sum()`](https://pytorch.org/docs/stable/torch.html#torch.sum): returns sum of all elements in a tensor

3. Single layer neural networks solution
[torch.mm()](https://pytorch.org/docs/stable/torch.html#torch.mm): matrix multiplication
for matrix multiplications, the number of columns in the first tensor must equal to the number of rows in the second column.
- `weights.reshape(a, b)` will return a new tensor with the same data as `weights` with size `(a, b)` sometimes, and sometimes a clone, as in it copies the data to another part of memory.
- `weights.resize_(a, b)` returns the same tensor with a different shape. However, if the new shape results in fewer elements than the original tensor, some elements will be removed from the tensor (but not from memory). If the new shape results in more elements than the original tensor, new elements will be uninitialized in memory. Here I should note that *the underscore at the end of the method denotes that this method is performed **in-place**.* Here is a great forum thread to [read more about in-place operations](https://discuss.pytorch.org/t/what-is-in-place-operation/16244) in PyTorch.
- `weights.view(a, b)` will return a new tensor with the same data as `weights` with size `(a, b)`.

usually use `.view()`, but any of the three methods will work for this.

use `tensor.shape` to debug common bugs




4. 
parameters: weights and biases 
hyperparameters: number of hidden units (parameter of the network) (不是多少层, 而是每层有多少个 unit)

5. 
pytorch: smooth transition between numpy arrays and torch tensors
```python
import numpy as np
a = np.random.rand(4,3)
# numpy to tensor
b = torch.from_numpy(a)
# tensor to numpy
b.numpy()
```
The memory is shared between the Numpy array and Torch tensor, so if you change the values in-place of one object, the other will change as well.

6. 

7. 
quickly flatten a tensor without having to know what the 2nd dimension has to be:
```python
inputs = image.view(image.shape[0], -1)
```
when implementing softmax, make sure you each row is devided by the corrsponding sum, which means you have to reshape the denominator:
```python
def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)
```


8. 

building network

Create a network with 784 input units, a hidden layer with 128 units and a ReLU activation, then a hidden layer with 64 units and a ReLU activation, and finally an output layer with a softmax activation as shown above. You can use a ReLU activation with the nn.ReLU module or F.relu function:
```python
## Your solution here
import torch.nn.functional as F

class Network(nn.Module):
# subclass from nn.Module
    def __init__(self):
        super().__init__()
        # run the init module to register all the layers and weights you will be creating and putting to this network later
        # if you don't do this then you won't be able to track what you are going to add to the nn

        # Inputs to hidden layer 1 linear transformation
        # it created an operation for the linear transformation
        self.fc1 = nn.Linear(784, 256)
        # hidden layer 1 to hidden layer 2 linear transformation
        self.fc2 = nn.Linear(256, 64)
        # Output layer, 10 units - one for each digit
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer with softmax activation
        x = F.softmax(self.fc3(x), dim=1)
        
        return x
```

### Activation functions
The only requirement is that for a network to approximate a non-linear function, the activation functions must be non-linear. 

- sigmoid activation function
- Tanh (hyperbolic tangent)
- ReLU (rectified linear unit) (In practice, the ReLU function is used almost exclusively as the activation function for hidden layers, simple and faster)


initialize weights and bias
```python
# below are just 2 examples

# Set biases to all zeros
model.fc1.bias.data.fill_(0)
# sample from random normal with standard dev = 0.01
model.fc1.weight.data.normal_(std=0.01)
```

`nn.sequential`
```python
# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
print(model)
```

look at the parameters of the model:
```python
print(model[0])
model[0].weight
```

or pass in an OrderedDict to name the individual layers and operations, instead of using incremental integers:
```python
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('output', nn.Linear(hidden_sizes[1], output_size)),
                      ('softmax', nn.Softmax(dim=1))]))
model

print(model[0])
print(model.fc1)
```


9. 
forward: get the loss
backward: get the graident and update weights & biases


You'll usually see the loss assigned to `criterion`. As noted in the last part, with a classification problem such as MNIST, we're using the softmax function to predict class probabilities. With a softmax output, you want to use cross-entropy as the loss. To actually calculate the loss, you first define the criterion then pass in the output of your network and the correct labels.
Something really important to note here. Looking at the documentation for nn.CrossEntropyLoss,
This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

The input is expected to contain scores for each class.
This means we need to pass in the raw output of our network into the loss, not the output of the softmax function. This raw output is usually called the logits or scores. We use the logits because softmax gives you probabilities which will often be very close to zero or one but floating-point numbers can't accurately represent values near zero or one (read more [here](https://docs.python.org/3/tutorial/floatingpoint.html)). It's usually best to avoid doing calculations with probabilities, typically we use log-probabilities.


10. 

It's more convenient to build the model with a `log-softmax` output using `nn.LogSoftmax` or `F.log_softmax` (documentation). Then you can get the actual probabilities by taking the exponential `torch.exp(output)`. With a `log-softmax` output, you want to use the negative log likelihood loss, `nn.NLLLoss` (documentation).
```python
 TODO: Build a feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1)
                     )

# TODO: Define the loss
criterion = nn.NLLLoss()

### Run this to check your work
# Get our data
images, labels = next(iter(trainloader))
# Flatten images
images = images.view(images.shape[0], -1)

# Forward pass, get our logits
logits = model(images)
# Calculate the loss with the logits and the labels
loss = criterion(logits, labels)

print(loss)
```
to sum up:
output of the model is log of softmax
<= softmax is the "propability" of each class predicted by the model
=> then use Negative Log Likelihood Loss function to calculate loss function.


### Autograd
Autograd works by keeping track of operations performed on tensors, then going backwards through those operations, calculating gradients along the way. To make sure PyTorch keeps track of operations on a tensor and calculates the gradients, you need to set `requires_grad = True` on a tensor. You can do this at creation with the `requires_grad` keyword, or at any time with `x.requires_grad_(True)`:

```python
x = torch.randn(2,2, requires_grad=True)
```

you can turn on or off gradients altogether with `torch.set_grad_enabled(True|False)`.

```python
# initialize a tensor
x = torch.randn(2,2, requires_grad=True)
y = x**2
z = y.mean()

## grad_fn shows the function that generated this variable
print(y.grad_fn) # power
print(z.grad_fn) # mean

# calculate gradient of z with respect to x
z.backward()

# check gradient for x
print(x.grad)
```

calculate the gradients for the parameters:
`loss.backward()`

we use `optimizer` to upgrade weights and bias using gradient descent. 

When you do multiple backwards passes with the same parameters, the gradients are accumulated. This means that you need to zero the gradients on each training pass or you'll retain gradients from previous training batches:
```python
optimizer.zero_grad()
```

**epoch**: one pass through the entire dataset

one training batch:
- training pass: calculate the loss, 
- backwards pass: update the weights.

A general learning step with PyTorch:
- Make a forward pass through the network 
- Use the network output to calculate the loss
- Perform a backward pass through the network with `loss.backward()` to calculate the gradients
- Take a step with the optimizer to update the weights

put all the ~~thing~~ together and call it a Presidents Day 2019:
```python
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim


# DATA PREPARATION
## Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ])
## Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


# MODEL TRAINING
## Build a feed-forward network that returns the log-softmax as the output 
## ... and calculate the loss using the negative log likelihood loss.
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
## use Stochastic Gradient Descent as optimizer that updates weights and bias for each iteration
optimizer = optim.SGD(model.parameters(), lr=0.003)

## train the model for 5 epochs
epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        ## Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        # Clear the gradients, do this because gradients are accumulated
        optimizer.zero_grad()
        ## make prediction
        output = model(images)
        ## calculate negative log likelihood loss
        loss = criterion(output, labels)
        ## use autograd to calculate gradient
        loss.backward()
        ## update weights and biases using the gradient calculated
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")
        

## CHECK MODEL PREDICTION AFTER TRAINING

%matplotlib inline
import helper

images, labels = next(iter(trainloader))

img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)

# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
helper.view_classify(img.view(1, 28, 28), ps)
```




link: 
- [yeezhu/SPN.pytorch/blob/master/environment.yml](https://github.com/yeezhu/SPN.pytorch/blob/master/environment.yml)
- [PyTorch Documentations](https://pytorch.org/docs/stable/index.html#pytorch-documentation)
- [Deep Learning With PyTorch](https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad)
- [PyTorch Basics: Tensors and Gradients](https://medium.com/jovian-io/pytorch-basics-tensors-and-gradients-eb2f6e8a6eee)

