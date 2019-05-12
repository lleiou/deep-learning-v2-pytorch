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


above doesn't really work, what really work is creating a conda environment, please refer to `setup/environment.yml`, when you use it for the first time, create virtual environment first by running `bash ./setup/setup.sh` from the root folder. in the future, activate `pytorch` virtual environment every time you need to work on the projects in this course.

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
when implementing softmax, make sure each row is devided by the corrsponding sum, which means you have to reshape the denominator:
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
(注意是 entire dataset, 而不是data set的一部分走个来回. 教程中是对 trainloader 里面的 images 进行循环的, 每次循环就对64个图像做一次 feed forward 和 back propagation, 更新一下 model 的 weight 和 bias, 但是这不算一个 epoch, 因为整个 data set 还没有遍历完, 下次循环就要开始对接下来 64 个图像 (这里的 images 是 batch 形式的) 再来一次一样的, 再更新模型, 直到所有 images 都跑了一遍, 才算一个 epoch)

one training batch:
- training pass: calculate the loss, 
- backwards pass: update the weights.

🌟 A general learning step with PyTorch:
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

to sum up:



14.
Performance metrics:
- accuracy, the percentage of classes the network predicted correctly
- [Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context))
- top-5 error rate

> The network learns the training set better and better, resulting in lower training losses. However, it starts having problems generalizing to data outside the training set leading to the validation loss increasing. The ultimate goal of any deep learning model is to make predictions on new data, so we should strive to get the lowest validation loss possible. 


15. 


Two solutions to overfitting:
- early-stopping: use the version of the model with the lowest validation loss, here the one around 8-10 training epochs. In practice, you'd save the model frequently as you're training then later choose the model with the lowest validation loss.
- dropout: randomly drop input units. This forces the network to share information between weights, increasing it's ability to generalize *to* new data.
```python
    ...        
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)
        # this is definining a method for the classs using an existing function
```

During training we want to use dropout to prevent overfitting, but during inference we want to use the entire network. So, **we need to turn off dropout during validation, testing, and whenever we're using the network to make predictions.** 

#### The general pattern for the validation loop
turn off gradients \
-> set the model to evaluation mode where the dropout probability is 0 (use `model.eval()`)\
-> calculate the validation loss and metric \
-> then set the model back to train mode (use `model.train()`)



17. saving an loadiNG models
The parameters for PyTorch networks are stored in a model's state_dict. We can see the state dict contains the weight and bias matrices for each of our layers.
```python
print("Our model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())

# save state dict
torch.save(model.state_dict(), 'checkpoint.pth')

# load state dict
model.load_state_dict(state_dict)
```
Loading the state dict works only if the model architecture is exactly the same as the checkpoint architecture. If I create a model with a different architecture, this fails. This means we need to rebuild the model exactly as it was when trained. Information about the model architecture needs to be saved in the checkpoint, along with the state dict. To do this, you build a dictionary with all the information you need to compeletely rebuild the model.


# I DON'T GET IT, WHAT'S THE POINT OF REBUILDING THE MODEL HERE?
# THERE COULD BE SOME CONFUSION IN THE LAST 4 CELLS
https://github.com/udacity/deep-learning-v2-pytorch/issues/84


would be better to read 

[SAVING AND LOADING MODELS | PyTorch](https://pytorch.org/tutorials/beginner/saving_loading_models.html)



18. loading image data

### Create ImageFolder
in the image directory, each class has it's own directory. The images are then labeled with the class taken from the directory name. 
```python
from torchvision import datasets, transforms
dataset = datasets.ImageFolder('path/to/data', transform=transform)
```

### Transforms
[transform](http://pytorch.org/docs/master/torchvision/transforms.html) is a list of processing steps built with the transforms module from `torchvision`.  the images are different sizes but we'll need them to all be the same size for training. Typically you'll combine these transforms into a pipeline with `transforms.Compose()`, which accepts a list of transforms and runs them in sequence. It looks something like this to scale, then crop, then convert to a tensor:
```python
transform = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()])
```

### Data Loaders
With the ImageFolder loaded, you have to pass it to a DataLoader. The DataLoader takes a dataset (such as you would get from ImageFolder) and returns batches of images and the corresponding labels. You can set various parameters like the batch size and if the data is shuffled after each epoch. 
也就是给每个图像配上 label 并打乱顺序, 为 train model 做好准备

```python
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
# shuffle=True prevents you from learning through the same order each time
```

Here `dataloader` is a [generator](https://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/). To get data out of it, you need to loop through it or convert it to an iterator and call `next()`.

```python
# Looping through it, get a batch on each loop 
for images, labels in dataloader:
    pass

# Get one batch
images, labels = next(iter(dataloader))

# show picture
import helper
helper.imshow(images[0], normalize=False)
```


### Data Augmentation
introduce randomness in the input data itself: rotate, mirror, scale, and/or crop.

```python
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])])
```

You'll also typically want to normalize images with `transforms.Normalize`. You pass in a list of means and list of standard deviations, then the color channels are normalized like so

```input[channel] = (input[channel] - mean[channel]) / std[channel]```

Normalizing helps keep the network work weights near zero which in turn makes backpropagation more stable. **Without normalization, networks will tend to fail to learn.**

You can find a list of all [the available transforms here](http://pytorch.org/docs/0.3.0/torchvision/transforms.html). When you're testing however, you'll want to use images that aren't altered (except you'll need to normalize the same way). So, for validation/test images, you'll typically just resize and crop. 因为 augmentation 只是





22. tips, tricks, and others
Watch those shapes
In general, you'll want to check that the tensors going through your model and other code are the correct shapes. Make use of the `.shape` method during debugging and development.

A few things to check if your network isn't training appropriately
Make sure you're clearing the gradients in the training loop with `optimizer.zero_grad()`. If you're doing a validation loop, be sure to set the network to evaluation mode with `model.eval()`, then back to training mode with `model.train()`.

CUDA errors
Sometimes you'll see this error:
```
RuntimeError: Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor for argument #1 ‘mat1’
```
You'll notice the second type is `torch.cuda.FloatTensor`, this means it's a tensor that has been moved to the GPU. It's expecting a tensor with type `torch.FloatTensor`, no .cuda there, which means the tensor should be on the CPU. **PyTorch can only perform operations on tensors that are on the same device, so either both CPU or both GPU**. If you're trying to run your network on the GPU, check to make sure you've moved the model and all necessary tensors to the GPU with `.to(device)` where `device` is either "cuda" or "cpu".









link: 
- [yeezhu/SPN.pytorch/blob/master/environment.yml](https://github.com/yeezhu/SPN.pytorch/blob/master/environment.yml)
- [PyTorch Documentations](https://pytorch.org/docs/stable/index.html#pytorch-documentation)
- [Deep Learning With PyTorch](https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad)
- [PyTorch Basics: Tensors and Gradients](https://medium.com/jovian-io/pytorch-basics-tensors-and-gradients-eb2f6e8a6eee)

