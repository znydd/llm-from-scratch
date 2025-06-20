import torch
import torch.nn as nn
import torch.optim as optim

class BasicFFN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size ):
        super(BasicFFN, self).__init__()
        self.connection_1 = nn.Linear(input_size, hidden_size_1)
        self.connection_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.connection_3 = nn.Linear(hidden_size_2, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, inp_tensor):
        #1st layer
        out = self.connection_1(inp_tensor)
        out = self.relu(out)

        #2nd layer
        out = self.connection_2(out)
        out = self.relu(out)

        #output layer
        out = self.connection_3(out)
        return out

x_train = torch.randn(100, 10)
y_train = torch.randint(0,2, (100, 1)).float()

input_dim = 10
hidden_dim1 = 32
hidden_dim2 = 16
output_dim = 1

model = BasicFFN(input_dim, hidden_dim1, hidden_dim2, output_dim)
print("Model Arch")
print(model)

loss_fn = nn.BCEWithLogitsLoss()
learning_rate = 0.001
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

num_epoch = 100

print("\n Starting Training")
for epoch in range(num_epoch):
    outputs = model(x_train)
    loss = loss_fn(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}')
        
print("\nTraining Done")


print("\n--- Testing the Model ---")
X_test = torch.randn(5, 10) 

model.eval()

with torch.no_grad():
    test_outputs = model(X_test)
    probabilities = torch.sigmoid(test_outputs)
    predicted_classes = probabilities.round()

    print("Test Data Input:")
    print(X_test)
    print("\nModel Output (Logits):")
    print(test_outputs)
    print("\nProbabilities:")
    print(probabilities)
    print("\nPredicted Classes (0 or 1):")
    print(predicted_classes)