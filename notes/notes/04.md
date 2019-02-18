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






link: 
- [yeezhu/SPN.pytorch/blob/master/environment.yml](https://github.com/yeezhu/SPN.pytorch/blob/master/environment.yml)
- [PyTorch Documentations](https://pytorch.org/docs/stable/index.html#pytorch-documentation)
- [Deep Learning With PyTorch](https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad)
- [PyTorch Basics: Tensors and Gradients](https://medium.com/jovian-io/pytorch-basics-tensors-and-gradients-eb2f6e8a6eee)