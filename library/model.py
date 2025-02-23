import torch.nn as nn
import torch 

class NNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NNModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight) #well-balanced weight distribution
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

        self.activation_cache = []


    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            tuple: (output, hidden_activations)
                output: Model output
                hidden_activations: Activations of the hidden layer
        """
        # Flatten the input
        x = x.view(x.size(0), -1)

        # Hidden layer with ReLU activation
        hidden = self.fc1(x)
        hidden_activations = self.relu(hidden)

        #self.training is a boolean attribute to track these things
        if not self.training:
            self.activation_cache.append(hidden_activations.detach().cpu().numpy())

        # Output layer
        output = self.fc2(hidden_activations)

        return output, hidden_activations

    def get_weights(self):
        """
        Get the weights of the model.

        Returns:
            tuple: (fc1_weights, fc2_weights)
                fc1_weights: Weights of the first fully connected layer
                fc2_weights: Weights of the second fully connected layer
        """
        return self.fc1.weight.data.cpu().numpy(), self.fc2.weight.data.cpu().numpy()

