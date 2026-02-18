# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Problem Statement

Regression problems aim to predict a continuous numerical value based on input features. Traditional regression models may fail to capture complex non-linear relationships.
A Neural Network Regression Model uses multiple layers of neurons to learn these non-linear patterns and improve prediction accuracy.

## Neural Network Model

<img width="869" height="713" alt="image" src="https://github.com/user-attachments/assets/bec82200-a092-45c8-a3f1-f988e7eac296" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: NETHRAA N
### Register Number: 212224040217
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
model = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train_model(nethraa_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = nethraa_brain(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses

```
## Dataset Information

<img width="196" height="351" alt="image" src="https://github.com/user-attachments/assets/81525732-5cf7-433b-9b70-56cd7cf89865" />


## OUTPUT

### Training Loss Vs Iteration Plot

<img width="754" height="606" alt="image" src="https://github.com/user-attachments/assets/06790a14-e6f1-4642-9ea1-ab1d53f4c42e" />


### New Sample Data Prediction

<img width="840" height="527" alt="image" src="https://github.com/user-attachments/assets/6496aedf-505e-4e63-89e8-98133a8d42f1" />



## RESULT

The neural network regression model was successfully developed and trained.
