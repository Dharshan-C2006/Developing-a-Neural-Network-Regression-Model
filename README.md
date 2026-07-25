# Developing a Neural Network Regression Model
### NAME : C Dharshan
### REG NO : 212224230059
## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

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

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:

### Register Number:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv( r"C:\Users\LENOVO\Documents\deeplearn\envi\Exp-1.csv")

x = df[["Input"]].values
y = df[["Output"]].values

xt, xst, yt, yst = train_test_split( x,y, test_size=0.2, random_state=52)

scaler_x = MinMaxScaler()

xt = scaler_x.fit_transform(xt)
xst = scaler_x.transform(xst)

xt1 = torch.FloatTensor(xt)
xst1 = torch.FloatTensor(xst)

yt1 = torch.FloatTensor(yt)
yst1 = torch.FloatTensor(yst)

class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential( nn.Linear(1,16), nn.ReLU(), nn.Linear(16,8), nn.ReLU(),  nn.Linear(8,1) )
    def forward(self,x):
        return self.network(x)

model = NeuralNet()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 2000
loss_history = []
for i in range(epochs):

    optimizer.zero_grad()
    output = model(xt1)
    loss = criterion( output, yt1  )
    loss.backward()

    optimizer.step()
    # Save loss
    loss_history.append(loss.item())
    if i % 100 == 0:
        print( f"Iteration {i}/{epochs} Loss: {loss.item()}"  )
model.eval()
input_value = torch.FloatTensor([[16]])
scaled_input = torch.FloatTensor( scaler_x.transform(input_value))
with torch.no_grad():
    prediction = model( scaled_input   )
print("\nPrediction for Input 16:")
print(prediction.item())

plt.figure(figsize=(8,5))
plt.plot( range(epochs), loss_history)
plt.xlabel("Iterations")
plt.ylabel( "Training Loss")
plt.title( "Training Loss vs Iteration")
plt.grid(True)


plt.show()


```

### Dataset Information

<img width="229" height="331" alt="image" src="https://github.com/user-attachments/assets/d50709e0-5d25-4f82-b93b-36bf83353daa" />


### OUTPUT

<img width="1600" height="1000" alt="image" src="https://github.com/user-attachments/assets/765532b6-d01e-46a9-8294-426f38f797e5" />


### Training Loss Vs Iteration Plot

<img width="1600" height="1000" alt="image" src="https://github.com/user-attachments/assets/b16a7e43-5169-438d-bb0a-4330f3bd3a7f" />


### New Sample Data Prediction

<img width="1600" height="1000" alt="image" src="https://github.com/user-attachments/assets/bb062446-4469-4f7b-9974-891c5ebcc16a" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
