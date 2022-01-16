# Deep-learning-school
These are my homeworks from Deep Learning School(DLS)

## Gradient optimization

Deep-learning-school / HW_gradient

**Main goal** - working with the concept of derivative and gradient and writing gradient descent and its variations.

***Tasks 1-3.***
In this task I had to calculate the derivative of different functions

***Task 4.***
In this assignment I needed to write a function for approximate calculation of the derivative of a function of one variable, and then check the resulting answer with the result of the code below.

![изображение](https://user-images.githubusercontent.com/78569587/149675078-c067ee9e-42e3-4841-93c9-d0d287761017.png)

*Task 5.*
Search for the minimum of functions using gradient descent and a visual representation of it.

![изображение](https://user-images.githubusercontent.com/78569587/149674952-5b9bacc4-090e-453e-96e2-64a4a713fecd.png)

*Task 6.*

Search for a global minimum for each function.

![изображение](https://user-images.githubusercontent.com/78569587/149675090-d1fbf33a-7752-4eaa-b8f3-bd4ddb45956e.png)

*Tasks 7-8.*
Calculate function gradients using code.

*Task 9.*
Writing a function that looks for the global minimum of the function

_____________________________________________________________________________________________

## Classification of the Simpsons(Competition on Kaggle)

Deep-learning-school / HW_Classification_Simpsones

**Main goal** - improving the neural network and increase the score.

Initially, I was given a very simple network where there was a low score.

Steps:

1. **I applied “ResNet18”. I've unfrozen last convolutional block and built new classifier layer**

![изображение](https://user-images.githubusercontent.com/78569587/149675106-010f4e8f-0008-4357-943a-02b788bf174e.png)

1. **I made some augmentantions(such as crop, random zoom, random rotation etc.)**
2. **I used Imbalanced dataset processing (WeightedRandomSampler)**

```python
from torch.utils.data import WeightedRandomSampler
```

```python
def train(train_files, val_files, model, epochs, batch_size, opt=None, sheduler=None):
  class_sample_count = counts
  weight = 1. / class_sample_count
  samples_weight = weight
  samples_weight = torch.from_numpy(samples_weight)
  sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
....
```

1. **As for me, it was necessary to use optimizer - AdamW**

```python
# Observe that all parameters are being optimized
optimizer_ft = optim.AdamW(model_ft.parameters())
```

1. **At the last, I used LearningRateScheduler**

```python
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
```

```python
def train(..., sheduler=None):
...
	sheduler.step()
```

_____________________________________________________________________________________________

## Fully connected & Convolutional Neural Networks

Deep-learning-school / HW_CNN

**Main goal** - training on building neural networks using the Pwtorch library on different datasets.

### Dataset moons

1. I uploaded the data using PyTorch
2. I wrote the module on PyTorch, which realize “logits = XW+b”

```python
class LinearRegression(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(out_features, in_features)
        )  # changed the str - added in и out features
        self.bias = bias
        if bias:
            self.bias_term = nn.Parameter(torch.randn(out_features))  # added Parameter

    def forward(self, x):
        x = x @ self.weights.t()
        if self.bias:
            x += self.bias
        return x
```

1. I realized learning cycles
2. Implemented predict and calculated accuracy on test

```python
@torch.no_grad()
def predict(dataloader, model):
    model.eval()
    predictions = np.array([])
    for x_batch, _ in dataloader:

        preds = preds = torch.argmax(
            x_batch @ W, dim=1
        )  # YOUR CODE. Compute predictions
        predictions = np.hstack((predictions, preds.numpy().flatten()))
    return predictions.flatten()
```

```python
from sklearn.metrics import accuracy_score

# Computed total accuracy
acc = 0
acc += (preds == y_batch).cpu().numpy().mean()

acc
```

### Dataset MNIST

I needed to create a fully connected neural network using the Sequential class. The network should consist of:

- Flattening of a matrix into a vector (nn.Flatten);
- Two hidden layers of 128 neurons with nn activation.ELU;
- Output layer with 10 neurons.

Set the loss for training (cross-entropy).

1. Set a loss for training

```python
criterion = nn.CrossEntropyLoss()  # selected a loss function
optimizer = torch.optim.Adam(model.parameters())

loaders = {"train": train_dataloader, "valid": valid_dataloader}
```

1. I have completed the training cycle
2. I have tested different activation functions

```python
# YOUR CODE. Do the same thing with other activations (it's better to wrap into a function that returns a list of accuracies)

def test_activation_function(activation):
    model = kotik(activation)
    criterion = nn.CrossEntropyLoss()  # selected a loss function
    optimizer = torch.optim.Adam(model.parameters())
    max_epochs = 10  # to brief overview result
    accuracy = {"train": [], "valid": []}
    model.to(device)
    for epoch in range(max_epochs):
        for k, dataloader in loaders.items():
            epoch_correct = 0
            epoch_all = 0
            for x_batch, y_batch in dataloader:
                if k == "train":
                    model.train()
                    # YOUR CODE. Set model to ``train`` mode and calculate outputs. Don't forget zero_grad!
                    # X_batch, y_batch = batch
                    optimizer.zero_grad()

                    outp = model(x_batch.to(device))
                    loss = criterion(outp.to(device), y_batch.to(device))
                    loss.backward()
                    optimizer.step()
                else:
                    # YOUR CODE. Set model to ``eval`` mode and calculate outputs
                    model.eval()
                    with torch.no_grad():
                        outp = model(x_batch.to(device))

                preds = outp.argmax(-1)
                correct = (preds.cpu().detach() == y_batch).sum()  # YOUR CODE GOES HERE
                all = len(y_batch)
                epoch_correct += correct.item()
                epoch_all += all
                # if k == "train":
                #     loss = criterion(outp, y_batch)
                #     loss.backward()
                #     optimizer.step()
                # YOUR CODE. Calculate gradients and make a step of your optimizer
            if k == "train":
                print(f"Epoch: {epoch+1}")
            print(f"Loader: {k}. Accuracy: {epoch_correct/epoch_all}")
            accuracy[k].append(epoch_correct / epoch_all)
    return accuracy["valid"]
```

1. Calculation of accuracy.

At the end of the notebook, it was necessary to work with CNN and change the original picture:

![изображение](https://user-images.githubusercontent.com/78569587/149675113-a9830ce3-5be4-4024-8902-2aeeb7aed8e3.png)

I realized this:

```python
img_t = torch.from_numpy(RGB_img).type(torch.float32).unsqueeze(0)
kernel = (
    torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    .reshape(1, 1, 3, 3)
    .type(torch.float32)
)

kernel = kernel.repeat(3, 3, 1, 1)
img_t = img_t.permute(0, 3, 1, 2)  # [BS, H, W, C] -> [BS, C, H, W]
img_t = nn.ReflectionPad2d(1)(img_t)  # Pad Image for same output size

result = F.conv2d(img_t, kernel)[0]
```

```python
plt.figure(figsize=(12, 8))
result_np = result.permute(1, 2, 0).numpy() / 256 / 3

plt.imshow(result_np)
plt.show()
```

And that's what happened:

![изображение](https://user-images.githubusercontent.com/78569587/149675129-75a1c70e-490a-4dbd-9b1f-268d42cc96da.png)

At the very end of the file I needed to implement LeNet:

- 3x3 convolution (1 card at the input, 6 at the output) with ReLU activation;
- MaxPooling-a 2x2;
- 3x3 convolution (6 cards at the input, 16 at the output) with ReLU activation;
- MaxPooling-a 2x2;
- Flattening (nn.Flatten);
- Fully connected layer with 120 neurons and ReLU activation;
- Fully connected layer with 84 neurons and ReLU activation;
- Output layer of 10 neurons.

Realization:

```python
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Flatten = nn.Flatten()
        self.fc1 = nn.Linear(400, 120)  # !!!
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x = #YOUR CODE. Apply layers created in __init__.
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # print(x.shape)
        x = self.Flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

```python
model = LeNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

loaders = {"train": train_dataloader, "valid": valid_dataloader}
```

_____________________________________________________________________________________________

## Churn prediction

Deep-learning-school / churn_prediction

**Main goal** - learn to model the outflow of telecom company customers.

1. I uploaded the data
2. For numerical features, I built a histogram. For categorical, I counted the number of each value for each attribute.

```python
fig, ax = plt.subplots(4, 4, figsize = (15, 15))

for idx, key in enumerate(cat_cols):
  ax[idx//4, idx%4].bar(data[key].value_counts().index, data[key].value_counts()) # idx//4 - строка, idx%4 - столбец
plt.show()
```

```python
data.hist(column=num_cols, figsize=(15, 15), bins = 10)
plt.show()
```

![изображение](https://user-images.githubusercontent.com/78569587/149675153-40bff162-84e4-461f-8693-3944bbd639db.png)

![изображение](https://user-images.githubusercontent.com/78569587/149675160-8bbecd10-7e83-4965-aa1c-07114aec9ef1.png)

1. Application of linear models -
3.1. I processed the data so that Logistic Regression could be applied to it (I normalized numerical features, and encoded categorical ones using one-hot-encoding)
3.2. I tested different values of the hyperparameter C and chose the best one (you can test with=100, 10, 1, 0.1, 0.01, 0.001) according to the ROC-AUC metric.
2. Application of gradient boosting
I divided the sample into train/valid. I tested catboost with standard parameters. I tested different values of the parameters of the number of trees and learning_rate and chose the best combination according to the ROC-AUC metric.
3. I did cross-validation.

_____________________________________________________________________________________________
