import torch.nn as nn

class ANN(nn.Module):
    def __init__(self, hidden_dim, num_classes, num_layers=1, dropout=0.2):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()

        layers = []
        # First hidden layer uses LazyLinear to infer input dimension automatically.
        layers.append(nn.LazyLinear(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Repeat additional hidden layers if num_layers > 1
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Final output layer (doesn't include activation or dropout)
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.network(x)
        return x