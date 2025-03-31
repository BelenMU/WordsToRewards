import numpy as np
from config import COOR_MIN, COOR_MAX, NUM_SEGMENTS, HIDDEN_CELLS_REWARD, NUM_OBSERVATIONS
import string
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from config import COLORS
import torch.nn.functional as F

def create_reward_heatmap(normalized_test_positions, predictions, num_segments=NUM_SEGMENTS, target_pos = None):
    """
    Generates a two-dimensional heatmap of predicted rewards based on normalized test positions.
    The heatmap is divided into squares (segments) where each square's value is the mean
    prediction of positions that fall into it. Squares with no data are plotted in grey.

    Args:
        normalized_test_positions (list): An iterable of normalized positions (x, y),
            where both x and y are floats between 0 and 1.
        predictions (list): An iterable of predictions corresponding to each position.
            Each prediction is a single numerical value.
        num_segments (int): The number of segments along each side of the heatmap. This
            divides the heatmap into num_segments x num_segments squares.
        target_pos (tensor of size 2): The normalized position of the target. If it exisst
            it will appear as a star in the heatmap.

    Returns:
        np.ma.masked_array: A two-dimensional masked array with shape (num_segments, num_segments),
            where each element contains the mean prediction for the corresponding segment,
            and segments with no data are masked to appear grey when plotted.
    """
    # Create an array to hold the sum of predictions and the count of positions
    heatmap_sum = np.zeros((num_segments, num_segments))
    count = np.zeros((num_segments, num_segments))

    # Calculate the size of each segment
    segment_size = 1.0 / num_segments

    # Accumulate predictions within segments
    for position, prediction in zip(normalized_test_positions, predictions):
        # Convert tensor to numpy array if necessary
        if isinstance(position, torch.Tensor):
            position = position.numpy()
        
        # Determine the segment indices
        segment_x = int(position[0] // segment_size)
        segment_y = int(position[1] // segment_size)

        # Accumulate predictions and counts
        heatmap_sum[segment_y, segment_x] += prediction
        count[segment_y, segment_x] += 1

    # Avoid division by zero and apply where condition to handle 0 count
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap_mean = np.true_divide(heatmap_sum, count)
        heatmap_mean[count <= 10] = np.nan  # Assign NaN where count is low
        
    # Mask the NaN values (zero count areas)
    heatmap_masked = masked_array(heatmap_mean, np.isnan(heatmap_mean))
    
    # Get the current "hot" colormap
    hot_cmap = plt.cm.get_cmap('hot')
    # Create a new colormap from the hot colormap with grey as the bad (masked) color
    hot_cmap.set_bad('grey', 1.0)
    
    # Plot the heatmap with the modified colormap
    plt.imshow(heatmap_masked, cmap=hot_cmap, interpolation='nearest')
    plt.colorbar()
    
    if target_pos is not None:
        target_x = int(target_pos[0] // segment_size)
        target_y = int(target_pos[1] // segment_size)
        plt.scatter(target_x, target_y, s=250, c='cyan', marker='*', edgecolors='black', label='Target Position')
        
    # Flip Y axis so that it corresponds to video
    plt.gca().invert_yaxis()
    
    plt.show()

    return heatmap_masked


def extract_observation(features, num_segments=NUM_SEGMENTS):
    """
    Extracts relevant information from the input features and returns a list representing the environment's observation.

    Args:
        features (dict): Input features containing information about angle, fingertip position, target location,
                         and angular velocity.
        num_segments (int): Number of segments in which the grid has been divided. Default is set to NUM_SEGMENTS.

    Returns:
    list: A list of 'Reacher-v4' observations containing the following elements in order:
        1. Cosine and sine information derived from the angle in features.
        2. Target location coordinates.
        3. Angular velocity.
        4. Fingertip coordinates relative to the target location.
        5. A placeholder value (0 in this case).

    """
    # Get Cosine-Sine Information
    cosine_sine = _get_cosine_sine(features['angle'])
    
    # Get coordinates for fingertip
    input_string = features['fingertip_position']
    letters = ''.join(char for char in input_string if char.isalpha())
    numbers = int(''.join(char for char in input_string if char.isdigit()))
    coor_x = letter_to_coor(letters, num_segments)
    coor_y = _row_to_coor(numbers, num_segments)
    target_loc = features['target_location'] 
    coor_fingertip_target = [coor_x - target_loc[0], coor_y - target_loc[1]]
    
    return list(cosine_sine) + target_loc + features['angular_velocity'] + coor_fingertip_target + [0]

def _get_cosine_sine(degrees):
    """
    Convert angles in degrees to cosine and sine values.

    Args:
        degrees (list): A list containing the angles of the first and second joints in degrees.

    Returns:
        numpy.ndarray: A 1D numpy array with four entries representing
                cosine of the angle of the first joint, cosine of the angle of the second joint,
                sine of the angle of the first joint, and sine of the angle of the second joint.
    """

    # Extract angles in degrees
    angle_1_deg, angle_2_deg = degrees
    
    if angle_1_deg is None:
         angle_1_deg = 0
         angle_2_deg = 0
    
    # Convert degrees to radians
    angle_1_rad = np.radians(angle_1_deg)
    angle_2_rad = np.radians(angle_2_deg)

    # Calculate cosine and sine values
    cos_angle_1 = np.cos(angle_1_rad)
    cos_angle_2 = np.cos(angle_2_rad)
    sin_angle_1 = np.sin(angle_1_rad)
    sin_angle_2 = np.sin(angle_2_rad)

    return np.array([cos_angle_1, cos_angle_2, sin_angle_1, sin_angle_2])

def letter_to_coor(letter, num_segments=NUM_SEGMENTS):
    """
    Converts a letter representation to the corresponding horizontal coordinate.

    Args:
        letter (str): The letter(s) to be converted to a horizontal coordinate.
        num_segments (int, optional): The number of segments that were used to divide the width to create 
                            the grid. Default is defined by the global NUM_SEGMENTS variable.

    Returns:
        float: The horizontal coordinate corresponding to the input letter(s).
    """
    if len(letter) == 1:  # Single letter
        index = string.ascii_lowercase.index(letter)
    elif len(letter) == 2:  # Two-letter combination
        first_letter, second_letter = letter
        first_letter_index = string.ascii_lowercase.index(first_letter)
        second_letter_index = string.ascii_lowercase.index(second_letter)
        index = first_letter_index * 26 + second_letter_index
    else:
        raise ValueError("Invalid input format. Expected either a single letter or a two-letter combination.")

    normalized_coor = index / num_segments
    # Random noise so that training data is not always in the middle of the row/column
    noise = np.random.uniform(-1/NUM_SEGMENTS, 1/NUM_SEGMENTS, 1)
    coor = normalized_coor * (COOR_MAX - COOR_MIN) + COOR_MIN + noise
    return coor

def _row_to_coor(row, num_segments):
    """
    Maps a row number to the corresponding vertical coordinate.

    Args:
        row (int): The row number to be converted to a vertical coordinate.
        num_segments (int): The number of segments that was used to divide the height to create 
                            the grid.

    Returns:
        float: The vertical coordinate corresponding to the input row number.
    """
    if row < 1:
        print("row: ", row)
        print("row must be at least 1")
        row = 1 # Modify to 1
        #raise ValueError("row must be between 1 and num_segments")
        
    elif row > num_segments:
        print("row: ", row)
        print("row must be less than num_segments")
        row = num_segments # Modify to max. instead of raising an error
        #raise ValueError("row must be between 1 and num_segments")

    # Map the row number to a normalized coordinate in the range [0, 1]
    normalized_coor = (row - 1) / (num_segments - 1)
    # Random noise so that training data is not always in the middle of the row/column
    noise = np.random.uniform(-1/NUM_SEGMENTS, 1/NUM_SEGMENTS, 1) 
    coor = normalized_coor * (COOR_MAX - COOR_MIN) + COOR_MIN + noise
    
    return coor

def train_reward_model(my_reward_model, x_train, y_train, x_test, y_test, optimizer, criterion, num_epochs=1000, print_interval=50, show_figure=True):
    """
    Train the reward model and plot the training and test losses.

    Args:
        my_reward_model (torch.nn.Module): The reward model to train.
        x_train (torch.Tensor): Training input data.
        y_train (torch.Tensor): Training target data.
        x_test (torch.Tensor): Testing input data.
        y_test (torch.Tensor): Testing target data.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        criterion (torch.nn.modules.loss._Loss): The loss function.
        num_epochs (int, optional): The number of epochs to train for. Defaults to 1000.
        print_interval (int, optional): The interval at which loss is printed. Defaults to 500.
        show_figure (boolean, optional): Specifies wether to show the loss evolution. Defaults to True.
    
    Returns: 
        None
    """
    # Lists to store training and test losses
    train_losses = []
    test_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Forward pass on training data
        outputs_train = my_reward_model(x_train).squeeze() # I added the .squeeze() on 03/12
        loss_train = criterion(outputs_train, y_train)
    
        # Backward pass and optimization
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
    
        # Forward pass on test data
        my_reward_model.eval()
        with torch.no_grad():
            outputs_test = my_reward_model(x_test)
            loss_test = criterion(outputs_test, y_test)
        my_reward_model.train()
    
        # Save losses
        train_losses.append(loss_train.item())
        test_losses.append(loss_test.item())
        
        # Print loss every print_interval epochs
        if (epoch + 1) % print_interval == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss_train.item():.4f}, Test Loss: {loss_test.item():.4f}')
    my_reward_model.eval()
    if show_figure:
        # Plot training and test losses
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

def get_predictions_and_distances(my_reward_model, observations):
    """Generate predictions using a provided reward model, and calculate distance to target.

    This function takes as input a reward model and a list of observation tensors. It outputs a list of predictions
    generated by the model for each observation and calculates the Euclidean distances based on the x, y coordinates
    of the fingertip and target.

    Args:
        my_reward_model (torch.nn.Module): The reward model to make predictions with.
        observations (List[torch.Tensor]): A list of observation tensors for which to generate predictions.

    Returns:
        Tuple[List[float], List[float]]: A tuple containing two lists:
            - The first list contains prediction values obtained from the model for each input observation.
            - The second list contains calculated Euclidean distances from fingertip to target.

    """
    # Calculate the distances as the Euclidean distance between the fingertip and the target
    x_values = [output[8].item() for output in observations]
    y_values = [output[9].item() for output in observations]
    distances = [(x**2 + y**2)**0.5 for x, y in zip(x_values, y_values)]
    
    # Extract the prediction values from the tensors 
    prediction_values = get_predictions(my_reward_model, observations)
    return prediction_values, distances

def get_predictions(my_reward_model, observations):
    """Generate predictions using a provided reward model.

    This function takes as input a reward model and a list of observation tensors. It outputs a list of predictions
    generated by the model for each observation.

    Args:
        my_reward_model (torch.nn.Module): The reward model to make predictions with.
        observations (List[torch.Tensor]): A list of observation tensors for which to generate predictions.

    Returns:
        List[float]: prediction values obtained from the model for each input observation.

    """
    # Store predictions
    predictions = []
    # Make predictions using the model
    with torch.no_grad():
        for input_tensor in observations:
            prediction = my_reward_model(input_tensor.unsqueeze(0))
            predictions.append(prediction)
    # Extract the prediction values from the tensors 
    prediction_values = [pred.item() for pred_batch in predictions for pred in pred_batch]
    return prediction_values

def draw_RM_rewards_vs_distance(prediction_values, distances, title, num_bins = 80):
    """Creates a scatter plot of reward means against distances with error bars.

    This function divides the distances into bins, calculates the mean predicted reward
    for each bin, and plots these means against the midpoints of the bins with error bars
    representing the standard error of the means.

    Args:
        prediction_values (list or numpy.ndarray): An array of predicted values for the reward.
        distances (list or numpy.ndarray): An array of distances between fingertips and the target.
        title (str): The title of the plot.
        num_bins (int, optional): The number of bins to use for digitizing distances.
        
    Returns:
        None: This function only displays a plot and does not return any value.

    """
    bin_edges = np.linspace(min(distances), max(distances), num_bins + 1)
    
    # Digitize the distances - assign each distance to a bin
    bins = np.digitize(distances, bin_edges)
    
    # Group prediction values by bins
    binned_predictions = {}
    for bin_index, pred_value in zip(bins, prediction_values):
        if bin_index not in binned_predictions:
            binned_predictions[bin_index] = []
        binned_predictions[bin_index].append(pred_value)
    
    # Now calculate means and standard errors for each bin
    means = []
    standard_errors = []
    for bin_index in range(1, num_bins + 1):
        if bin_index in binned_predictions:
            bin_values = binned_predictions[bin_index]
            means.append(np.mean(bin_values))
            # Standard error is the standard deviation divided by the square root of the number of samples
            standard_errors.append(np.std(bin_values) / np.sqrt(len(bin_values)))
        else:
            # If there are no predictions in a bin, use NaN to skip it in the plot
            means.append(np.nan)
            standard_errors.append(np.nan)
    
    # Calculate the center of each bin for plotting
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    # Plotting the means with standard error bars
    plt.errorbar(bin_centers, means, yerr=standard_errors, fmt='o', ecolor='r', capthick=2)
    
    plt.xlabel('Distance between fingertip and target')
    plt.ylabel('Mean Predicted Reward')
    plt.title(title)
    plt.grid()
    plt.show()

def draw_histogram(all_values, max_bin, x_label, legend_entries, bin_number = 5):
    """
    Draws overlapping histograms for multiple sets of values on the same plot.

    Args:
        all_values (list of list of float): List of lists where each inner list
            contains numeric values for which histograms should be generated.
        max_bin (float): The maximum value for the binning of the histograms.
        x_label (str): The label for the x-axis of the histogram.
        legend_entries (list of str): Labels for the legend, one entry per set of values.
        bin_number (int): Number of bins in the historgram. Defaults to five.

    Returns:
        None. This function will produce an overlapping histogram plot using matplotlib.
    """
    total_points = [len(values) for values in all_values]
    # Define bins
    bin_width = max_bin / bin_number
    bins = np.arange(0, max_bin + 0.5 * bin_width, bin_width) 
    
    # Create a single subplot for overlapping histograms
    fig, ax = plt.subplots()
    colors = COLORS 
    # Plot histograms for each set of values with different transparency
    for i, values in enumerate(all_values):
        color = 'blue' if i == 0 else 'red'
        counts, edges, _ = ax.hist(values, bins=bins, color=colors[i], edgecolor='black', alpha=0.5, weights=np.ones_like(values) / total_points[i])
    
    # Set x-axis ticks to represent the bin centers
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    #ax.set_xticks(bin_centers)
    #ax.set_xticklabels([f'{(bins[j]+bins[j+1])/2:.1f}' for j in range(len(bins)-1)])
    #ax.set_xticklabels([f'{bins[j]:.1f}-{bins[j+1]:.1f}' for j in range(len(bins)-1)])
    ax.set_xticks(bin_centers[::2])  # Display every other tick
    ax.set_xticklabels([f'{(bins[j]+bins[j+1])/2:.1f}' for j in range(0, len(bins)-1, 2)])

    # Set y-axis to show percentages
    ax.set_yticklabels([f'{int(val*100)}%' for val in ax.get_yticks()])
    
    # Set common labels
    ax.set(xlabel=x_label, ylabel='Percentage of Instances')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.grid()
    plt.legend(legend_entries)
    plt.show()


class simple_NN_biased(nn.Module):
    def __init__(self, input_size, hidden_size, leaky_relu_slope=0.01, initial_bias=0):
        super(simple_NN_biased, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.output_layer = nn.Linear(hidden_size, 1)
        
        # Initialize output layer bias to a negative value for a 0-bias
        if initial_bias is not None:
            with torch.no_grad():  # Temporarily disables autograd
                self.output_layer.bias.fill_(initial_bias)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.leaky_relu(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

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

class simple_NN_withDropout(nn.Module):
    def __init__(self, input_size, hidden_size, leaky_relu_slope=0.01, dropout_prob=0.5):
        super(simple_NN_withDropout, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer added here
        self.output_layer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)  # Apply dropout after activation function
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

class Attention(nn.Module):
    def __init__(self, input_size, attention_size = 8):
        super(Attention, self).__init__()
        self.attention_w = nn.Parameter(torch.randn(attention_size, input_size))
        self.attention_u = nn.Parameter(torch.randn(attention_size, 1))

    def forward(self, x):
        u = torch.tanh(torch.matmul(x, self.attention_w.T))
        scores = torch.matmul(u, self.attention_u)
        # Ensure scores have the right shape to apply softmax over features
        if scores.dim() > 1:
            attn_scores = F.softmax(scores, dim=1)
        else:
            # If scores is 1D, softmax over the only dimension present
            attn_scores = F.softmax(scores, dim=0)
        weighted_input = x * attn_scores  # This is element-wise multiplication, broadcasting the attention scores over the features.

        return weighted_input  # This should still have the shape [batch_size, input_size].

class simple_NN_with_Attention(nn.Module):
    def __init__(self, input_size, attention_size, hidden_size, leaky_relu_slope=0.01):
        super(simple_NN_with_Attention, self).__init__()
        self.attention = Attention(input_size, attention_size)
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.layer2 = nn.Linear(hidden_size, hidden_size)  # additional layer for MLP
        self.output_layer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.attention(x)
        x = self.layer1(x)
        x = self.leaky_relu(x)
        x = self.layer2(x)
        x = self.leaky_relu(x)  # Reusing the same LeakyReLU for the hidden layers
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x
    
class SiameseNN(nn.Module):
    def __init__(self, input_size, hidden_size, leaky_relu_slope=0.01):
        super(SiameseNN, self).__init__()
        # Shared layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.output_layer = nn.Linear(hidden_size, 1)
        
    def forward_once(self, x):
        # Pass input through the network
        x = self.layer1(x)
        x = self.leaky_relu(x)
        x = self.output_layer(x)
        return x

    def forward(self, x1, x2):
        # Pass both inputs through the shared network
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)

        # Use subtraction to get a comparative score
        out = out2 - out1
            # Maps score to [0, 1] -> Closer to 0 (1) option x1 (x2) is preferred
        predicted_preference = torch.sigmoid(out)
        
        return predicted_preference
