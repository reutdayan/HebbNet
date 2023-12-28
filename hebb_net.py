import torch
from torch import nn

# Define model
class HebbNet(nn.Module):
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hebbian_weights = nn.Linear(input_layer_size, hidden_layer_size, False)
        self.classification_weights = nn.Linear(hidden_layer_size, output_layer_size, True)

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        z = self.hebbian_weights(x)
        z = self.relu(z)  # Apply ReLU activation after the Hebbian layer
        pred = self.classification_weights(z)
        pred = self.softmax(pred)
        return x,z,pred

# comptes the delta_w1 for the Hebbian layer (basic hebb rule + activation threshold)
class HebbRuleWithActivationThreshold(nn.Module):
    def __init__(self, hidden_layer_size=2000, input_layer_size=784):
        super().__init__()
        self.register_buffer('w1_activation_thresholds', torch.zeros((hidden_layer_size, input_layer_size)))
        self.t = 1

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        with torch.no_grad():
          activation = torch.matmul(z.T,x) #(2000,784) - matrix multipication

          if self.t==1:
              delta_w1 = activation
          else:
              delta_w1 = activation - (self.w1_activation_thresholds / (self.t-1))

          self.w1_activation_thresholds += activation
          self.t += 1
          return delta_w1
        
# returns delta_w1 after applying the gradient sparsity
def gradiant_sparsity(delta_w1, p, device):
  # Calculate the number of values to keep based on the percentile (p)
  num_values_to_keep = int( p * delta_w1.numel())

  # Find the top k values and their indices
  top_values, _ = torch.topk(torch.abs(delta_w1).view(-1), num_values_to_keep)
  threshold = top_values[-1]  # The threshold is the k-th largest value

  # Set values below the threshold to zero
  delta_w1 = torch.where(torch.abs(delta_w1) >= threshold, delta_w1, torch.tensor(0.0).to(device))
  return delta_w1