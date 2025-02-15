import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class simple_NN(nn.Module):
    def __init__(self, input_size, hidden_size, leaky_relu_slope=0.01):
        super(simple_NN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.leaky_relu(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x
        
def train_reward_model(my_reward_model, x_train, y_train, criterion, num_epochs=1000, print_interval=50, show_figure=True):
    """
    Train the reward model and plot the training and test losses.

    Args:
        my_reward_model (torch.nn.Module): The reward model to train.
        x_train (torch.Tensor): Training input data.
        y_train (torch.Tensor): Training target data.
        x_test (torch.Tensor): Testing input data.
        y_test (torch.Tensor): Testing target data.
        criterion (torch.nn.modules.loss._Loss): The loss function.
        num_epochs (int, optional): The number of epochs to train for. Defaults to 1000.
        print_interval (int, optional): The interval at which loss is printed. Defaults to 500.
        show_figure (boolean, optional): Specifies wether to show the loss evolution. Defaults to True.
    
    Returns: 
        None
    """
    optimizer = torch.optim.Adam(my_reward_model.parameters(), lr=0.001)
    
    # Lists to store training and test losses
    train_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Forward pass on training data
        outputs_train = my_reward_model(x_train).squeeze()
        loss_train = criterion(outputs_train, y_train)
    
        # Backward pass and optimization
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        my_reward_model.train()
    
        # Save losses
        train_losses.append(loss_train.item())
        
        # Print loss every print_interval epochs
        if (epoch + 1) % print_interval == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss_train.item():.4f}')
            
    my_reward_model.eval()
    if show_figure:
        # Plot training and test losses
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


def score_front_x(observation):
    # Elements to check (elements on x of the front side)
    indices_to_check = [9, 11, 13, 15, 17]
    color = 3 # Orange
    
    # Count how many of the specified elements are equal to color
    count_equal_to_color = sum(1 for index in indices_to_check if observation[index] == color)
    
    # Calculate the score proportionally
    score = count_equal_to_color / len(indices_to_check)
    
    return score

def score_front_japan(observation): 
    # All white except for the the middle square in red
    indices_white = [9, 10, 11, 12, 14, 15, 16, 17]
    white = 0 # 'W': 0, 'R': 1, 'B': 2, 'O': 3, 'G': 4, 'Y': 5
    index_red = 13
    red = 1
    
    # Count how many of the specified elements are equal to color
    count_equal_to_color = sum(1 for index in indices_white if observation[index] == white)
    if observation[index_red] == red:
        count_equal_to_color += 1
    
    # Calculate the score proportionally
    score = count_equal_to_color / 9
    
    return score

def score_front_italy(observation): 
    # Three vertical lines in G, W and R
    indices_green = [9, 12, 15]
    green = 4 # 'W': 0, 'R': 1, 'B': 2, 'O': 3, 'G': 4, 'Y': 5
    indices_white = [10,13, 16]
    white = 0 
    indices_red = [11, 14, 17]
    red = 1
    
    # Count how many of the specified elements are equal to color
    count_equal_to_green = sum(1 for index in indices_green if observation[index] == green)
    count_equal_to_white = sum(1 for index in indices_white if observation[index] == white)
    count_equal_to_red = sum(1 for index in indices_red if observation[index] == red)
    
    # Calculate the score proportionally
    score = (count_equal_to_green + count_equal_to_white + count_equal_to_red) / 9
    
    return score

def calculate_ensemble_reward(ensemble, obs_tensor):
    # Aggregate predictions by averaging over the ensemble
    rewards = [model.forward_once(obs_tensor).detach() for model in ensemble]
    avg_reward = torch.mean(torch.stack(rewards), dim=0)
    return avg_reward.numpy()