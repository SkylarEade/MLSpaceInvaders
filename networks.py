import numpy as np


class CNNBackbone:
    def __init__(self):
        # Convolution 1: 32 filters, 4 input channels, 8x8 kernel
        self.conv1_weights = np.random.randn(32,4,8,8) * (np.sqrt(2.0 / (4*8*8)))
        self.conv1_bias = np.zeros((32)) 
        # Convolution 2: 64 filters 32 input channels, 4x4 kernel
        self.conv2_weights = np.random.randn(64,32,4,4) * (np.sqrt(2.0 / (32*4*4)))
        self.conv2_bias = np.zeros((64))
        # Convolution 3: 64 filters, 64 input channels, 3x3 kernel
        self.conv3_weights = np.random.randn(64,64,3,3) * (np.sqrt(2.0 / (64*3*3)))
        self.conv3_bias = np.zeros((64))

    def conv2d(self, input, weights, bias, stride):
        """2D convolution"""
        batch, in_channels, in_height, in_width = input.shape
        out_channels, _, kernel_h, kernel_w = weights.shape
        # Calculate the output dimensions
        out_height = (in_height - kernel_h) // stride + 1
        out_width = (in_width - kernel_w) // stride + 1
        output = np.zeros((batch, out_channels, out_height, out_width))
        for b in range(batch):
            for c_out in range(out_channels):
                for y in range(out_height):
                    for x in range(out_width):
                        y_start = y * stride
                        x_start = x * stride
                        patch = input[b, :, y_start:y_start+kernel_h, x_start:x_start+kernel_w]
                        output[b, c_out, y, x] = np.sum(patch * weights[c_out]) + bias[c_out]
        return output
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def forward(self, state):
        if len(state.shape) == 3: # Handles the cases of single states
            state = state[np.newaxis, ...] # provides state a batch dimension
        z = self.conv2d(state, self.conv1_weights, self.conv1_bias, stride=4)
        z = self.relu(z)
        z = self.conv2d(z, self.conv2_weights, self.conv2_bias, stride=2)
        z = self.relu(z)
        z = self.conv2d(z, self.conv3_weights, self.conv3_bias, stride=1)
        z = self.relu(z)
        features = z.reshape(z.shape[0], -1)
        return features
    

class Actor():
    """Takes in the current environment (state) and determines the best action to take from there"""
    def __init__(self, feature_dim, num_actions):
        self.fc1_weights = np.random.randn(feature_dim, 512) * (np.sqrt(2.0 / 512)) # 512 is EXPERIMENTAL {can be tweaked}
        self.fc1_bias = np.zeros((512))
        self.fc2_weights = np.random.randn(512, num_actions) * (np.sqrt(2.0 / num_actions))
        self.fc2_bias = np.zeros((num_actions))

    def relu(self, z):
        return np.maximum(0, z)
    
    def softmax(self, logits):
        # subtract max to avoid overflow. axis=-1 means across actions, keepdims=True preserves shape
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_shifted = np.exp(shifted)
        probs = exp_shifted / np.sum(exp_shifted, axis=-1,keepdims=True) # Calculates probabilities of logits
        return probs

    def log_softmax(self, logits):
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        log_probs = shifted - log_sum_exp
        
        return log_probs

    def forward(self, features):
        z = features @ self.fc1_weights + self.fc1_bias
        z = self.relu(z)

        logits = z @ self.fc2_weights + self.fc2_bias
        return logits

    def get_action_probs(self, features):
        logits = self.forward(features)
        probs = self.softmax(logits)
        return probs

    def get_log_probs(self, features, actions):
        logits = self.forward(features)
        all_log_probs = self.log_softmax(logits)
        batch_size = features.shape[0]
        log_probs = all_log_probs[np.arange(batch_size), actions]
        return log_probs


class Critic:
    def __init__(self):
        pass
    