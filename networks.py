import numpy as np


class CNNBackbone:
    def __init__(self):
        # Convolution 1: 32 filters, 4 input channels, 8x8 kernel
        self.conv1_weights = np.random.randn(32, 4, 8, 8) * np.sqrt(2.0 / (4 * 8 * 8))
        self.conv1_bias = np.zeros(32)
        # Convolution 2: 64 filters, 32 input channels, 4x4 kernel
        self.conv2_weights = np.random.randn(64, 32, 4, 4) * np.sqrt(2.0 / (32 * 4 * 4))
        self.conv2_bias = np.zeros(64)
        # Convolution 3: 64 filters, 64 input channels, 3x3 kernel
        self.conv3_weights = np.random.randn(64, 64, 3, 3) * np.sqrt(2.0 / (64 * 3 * 3))
        self.conv3_bias = np.zeros(64)

        self.cache = {}
        self.adam_m = {}
        self.adam_v = {}
        self.adam_t = 0
        
        param_names = [
            'conv1_weights', 'conv1_bias',
            'conv2_weights', 'conv2_bias',
            'conv3_weights', 'conv3_bias'
        ]
        
        for param_name in param_names:
            param = getattr(self, param_name)
            self.adam_m[param_name] = np.zeros_like(param)
            self.adam_v[param_name] = np.zeros_like(param)

    def im2col(self, input, kernel_h, kernel_w, stride):
        """Convert image patches to columns for efficient convolution"""
        batch, in_channels, in_height, in_width = input.shape
        out_height = (in_height - kernel_h) // stride + 1
        out_width = (in_width - kernel_w) // stride + 1
        
        # Create output array
        col = np.zeros((batch, in_channels, kernel_h, kernel_w, out_height, out_width))
        
        for y in range(kernel_h):
            y_max = y + stride * out_height
            for x in range(kernel_w):
                x_max = x + stride * out_width
                col[:, :, y, x, :, :] = input[:, :, y:y_max:stride, x:x_max:stride]
        
        # Reshape to (batch * out_h * out_w, in_channels * kernel_h * kernel_w)
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch * out_height * out_width, -1)
        return col, out_height, out_width

    def conv2d(self, input, weights, bias, stride):
        """Vectorized 2D convolution using im2col"""
        batch, in_channels, in_height, in_width = input.shape
        out_channels, _, kernel_h, kernel_w = weights.shape
        
        # im2col transformation
        col, out_height, out_width = self.im2col(input, kernel_h, kernel_w, stride)
        
        # Reshape weights: (out_channels, in_channels * kernel_h * kernel_w)
        weights_col = weights.reshape(out_channels, -1)
        
        # Matrix multiplication
        output = col @ weights_col.T + bias  # (batch*out_h*out_w, out_channels)
        
        # Reshape to proper output shape
        output = output.reshape(batch, out_height, out_width, out_channels)
        output = output.transpose(0, 3, 1, 2)  # (batch, out_channels, out_h, out_w)
        
        return output
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, dL_da, z):
        return dL_da * (z > 0).astype(np.float32)
    
    def col2im(self, col, input_shape, kernel_h, kernel_w, stride, out_height, out_width):
        """Convert columns back to image (for backward pass)"""
        batch, in_channels, in_height, in_width = input_shape
        
        # Reshape col back
        col = col.reshape(batch, out_height, out_width, in_channels, kernel_h, kernel_w)
        col = col.transpose(0, 3, 4, 5, 1, 2)  # (batch, in_channels, kernel_h, kernel_w, out_h, out_w)
        
        img = np.zeros(input_shape)
        
        for y in range(kernel_h):
            y_max = y + stride * out_height
            for x in range(kernel_w):
                x_max = x + stride * out_width
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
        
        return img

    def conv2d_backward(self, dL_doutput, input, kernel, stride):
        """Vectorized backward pass for convolution"""
        batch, out_channels, out_height, out_width = dL_doutput.shape
        _, in_channels, kernel_h, kernel_w = kernel.shape
        
        # Gradient w.r.t bias
        dL_dbias = np.sum(dL_doutput, axis=(0, 2, 3))
        
        # Reshape for matrix operations
        dL_doutput_reshaped = dL_doutput.transpose(0, 2, 3, 1).reshape(-1, out_channels)
        
        # im2col for input
        col, _, _ = self.im2col(input, kernel_h, kernel_w, stride)
        
        # Gradient w.r.t weights
        dL_dkernel = (dL_doutput_reshaped.T @ col).reshape(kernel.shape)
        
        # Gradient w.r.t input
        weights_col = kernel.reshape(out_channels, -1)
        dcol = dL_doutput_reshaped @ weights_col  # (batch*out_h*out_w, in_channels*kernel_h*kernel_w)
        dL_dinput = self.col2im(dcol, input.shape, kernel_h, kernel_w, stride, out_height, out_width)
        
        return dL_dinput, dL_dkernel, dL_dbias

    def forward(self, state, store_activations=False):
        if len(state.shape) == 3:
            state = state[np.newaxis, ...]
        
        z1 = self.conv2d(state, self.conv1_weights, self.conv1_bias, stride=4)
        a1 = self.relu(z1)
        z2 = self.conv2d(a1, self.conv2_weights, self.conv2_bias, stride=2)
        a2 = self.relu(z2)
        z3 = self.conv2d(a2, self.conv3_weights, self.conv3_bias, stride=1)
        a3 = self.relu(z3)
        features = a3.reshape(a3.shape[0], -1)
        
        if store_activations:
            self.cache = {
                'state': state,
                'z1': z1, 'a1': a1,
                'z2': z2, 'a2': a2,
                'z3': z3, 'a3': a3,
                'features': features
            }
        return features
    
    def backward(self, dL_dfeatures):
        state = self.cache['state']
        z1, a1 = self.cache['z1'], self.cache['a1']
        z2, a2 = self.cache['z2'], self.cache['a2']
        z3, a3 = self.cache['z3'], self.cache['a3']

        batch = a3.shape[0]
        dL_da3 = dL_dfeatures.reshape(batch, 64, 7, 7)

        dL_dz3 = self.relu_derivative(dL_da3, z3)
        dL_da2, dL_dW3, dL_db3 = self.conv2d_backward(dL_dz3, a2, self.conv3_weights, stride=1)

        dL_dz2 = self.relu_derivative(dL_da2, z2)
        dL_da1, dL_dW2, dL_db2 = self.conv2d_backward(dL_dz2, a1, self.conv2_weights, stride=2)
    
        dL_dz1 = self.relu_derivative(dL_da1, z1)
        dL_dstate, dL_dW1, dL_db1 = self.conv2d_backward(dL_dz1, state, self.conv1_weights, stride=4)

        gradients = {
            'conv1_weights': dL_dW1,
            'conv1_bias': dL_db1,
            'conv2_weights': dL_dW2,
            'conv2_bias': dL_db2,
            'conv3_weights': dL_dW3,
            'conv3_bias': dL_db3
        }
        return dL_dstate, gradients
    
    def update_weights(self, gradients, lr=3e-4, beta1=0.9, beta2=0.999, epsilon=1e-8, max_grad_norm=0.5):
        self.adam_t += 1
        for param_name, gradient in gradients.items():
            # Gradient clipping
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > max_grad_norm:
                gradient = gradient * max_grad_norm / grad_norm
            
            param = getattr(self, param_name)
            self.adam_m[param_name] = beta1 * self.adam_m[param_name] + (1 - beta1) * gradient
            self.adam_v[param_name] = beta2 * self.adam_v[param_name] + (1 - beta2) * (gradient ** 2)
            m_hat = self.adam_m[param_name] / (1 - beta1 ** self.adam_t)
            v_hat = self.adam_v[param_name] / (1 - beta2 ** self.adam_t)
            param -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
            setattr(self, param_name, param)


class Actor:
    """Policy network: determines action probabilities given state features"""
    def __init__(self, feature_dim, num_actions):
        self.fc1_weights = np.random.randn(feature_dim, 512) * np.sqrt(2.0 / feature_dim)
        self.fc1_bias = np.zeros(512)
        self.fc2_weights = np.random.randn(512, num_actions) * np.sqrt(0.01)  # Small init for policy head
        self.fc2_bias = np.zeros(num_actions)
        self.cache = {}
        self.adam_m = {}
        self.adam_v = {}
        self.adam_t = 0
        for param_name in ['fc1_weights', 'fc1_bias', 'fc2_weights', 'fc2_bias']:
            param = getattr(self, param_name)
            self.adam_m[param_name] = np.zeros_like(param)
            self.adam_v[param_name] = np.zeros_like(param)

    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(np.float32)
    
    def softmax(self, logits):
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_shifted = np.exp(shifted)
        probs = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
        return probs

    def log_softmax(self, logits):
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        return shifted - log_sum_exp
    
    def forward(self, features, store_activations=False):
        z1 = features @ self.fc1_weights + self.fc1_bias
        a1 = self.relu(z1)
        logits = a1 @ self.fc2_weights + self.fc2_bias
        
        if store_activations:
            self.cache = {
                'features': features,
                'z1': z1,
                'a1': a1,
                'logits': logits
            }
        return logits

    def get_log_probs(self, features, actions):
        logits = self.forward(features)
        all_log_probs = self.log_softmax(logits)
        batch_size = features.shape[0]
        log_probs = all_log_probs[np.arange(batch_size), actions]
        return log_probs
    
    def backward(self, dL_dlogits):
        features = self.cache['features']
        z1 = self.cache['z1']
        a1 = self.cache['a1']

        dL_dW2 = a1.T @ dL_dlogits
        dL_db2 = np.sum(dL_dlogits, axis=0)
        dL_da1 = dL_dlogits @ self.fc2_weights.T
        dL_dz1 = dL_da1 * self.relu_derivative(z1)
        dL_dW1 = features.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0)
        dL_dfeatures = dL_dz1 @ self.fc1_weights.T
        
        gradients = {
            'fc1_weights': dL_dW1,
            'fc1_bias': dL_db1,
            'fc2_weights': dL_dW2,
            'fc2_bias': dL_db2
        }
        return dL_dfeatures, gradients

    def update_weights(self, gradients, lr=3e-4, beta1=0.9, beta2=0.999, epsilon=1e-8, max_grad_norm=0.5):
        self.adam_t += 1
        for param_name, gradient in gradients.items():
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > max_grad_norm:
                gradient = gradient * max_grad_norm / grad_norm
                
            param = getattr(self, param_name)
            self.adam_m[param_name] = beta1 * self.adam_m[param_name] + (1 - beta1) * gradient
            self.adam_v[param_name] = beta2 * self.adam_v[param_name] + (1 - beta2) * (gradient ** 2)
            m_hat = self.adam_m[param_name] / (1 - beta1 ** self.adam_t)
            v_hat = self.adam_v[param_name] / (1 - beta2 ** self.adam_t)
            param -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
            setattr(self, param_name, param)


class Critic:
    """Value network: estimates state value"""
    def __init__(self, feature_dim):
        self.fc1_weights = np.random.randn(feature_dim, 512) * np.sqrt(2.0 / feature_dim)
        self.fc1_bias = np.zeros(512)
        self.fc2_weights = np.random.randn(512, 1) * np.sqrt(1.0)  # Larger init for value head
        self.fc2_bias = np.zeros(1)
        self.cache = {}
        self.adam_m = {}
        self.adam_v = {}
        self.adam_t = 0
        for param_name in ['fc1_weights', 'fc1_bias', 'fc2_weights', 'fc2_bias']:
            param = getattr(self, param_name)
            self.adam_m[param_name] = np.zeros_like(param)
            self.adam_v[param_name] = np.zeros_like(param)

    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(np.float32)
    
    def forward(self, features, store_activations=False):
        z = features @ self.fc1_weights + self.fc1_bias
        a = self.relu(z)
        value = a @ self.fc2_weights + self.fc2_bias
        
        if store_activations:
            self.cache = {
                'features': features,
                'z1': z,
                'a1': a,
                'value': value
            }
        return value
    
    def backward(self, dL_dvalue):
        features = self.cache['features']
        z1 = self.cache['z1']
        a1 = self.cache['a1']

        dL_dW2 = a1.T @ dL_dvalue
        dL_db2 = np.sum(dL_dvalue, axis=0)
        dL_da1 = dL_dvalue @ self.fc2_weights.T
        dL_dz1 = dL_da1 * self.relu_derivative(z1)
        dL_dW1 = features.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0)
        dL_dfeatures = dL_dz1 @ self.fc1_weights.T
        
        gradients = {
            'fc1_weights': dL_dW1,
            'fc1_bias': dL_db1,
            'fc2_weights': dL_dW2,
            'fc2_bias': dL_db2
        }
        return dL_dfeatures, gradients
    
    def update_weights(self, gradients, lr=3e-4, beta1=0.9, beta2=0.999, epsilon=1e-8, max_grad_norm=0.5):
        self.adam_t += 1
        for param_name, gradient in gradients.items():
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > max_grad_norm:
                gradient = gradient * max_grad_norm / grad_norm
                
            param = getattr(self, param_name)
            self.adam_m[param_name] = beta1 * self.adam_m[param_name] + (1 - beta1) * gradient
            self.adam_v[param_name] = beta2 * self.adam_v[param_name] + (1 - beta2) * (gradient ** 2)
            m_hat = self.adam_m[param_name] / (1 - beta1 ** self.adam_t)
            v_hat = self.adam_v[param_name] / (1 - beta2 ** self.adam_t)
            param -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
            setattr(self, param_name, param)


class ActorCritic:
    def __init__(self):
        self.cnn = CNNBackbone()
        self.actor = Actor(feature_dim=3136, num_actions=6)
        self.critic = Critic(feature_dim=3136)

    def forward(self, state, store_activations=False):
        """Single forward pass for both actor and critic"""
        features = self.cnn.forward(state, store_activations=store_activations)
        logits = self.actor.forward(features, store_activations=store_activations)
        value = self.critic.forward(features, store_activations=store_activations)
        return logits, value, features

    def get_action_and_value(self, state):
        """Get action (sampled), value, and log_prob for rollout collection"""
        logits, value, _ = self.forward(state, store_activations=False)
        probs = self.actor.softmax(logits)
        
        if len(probs.shape) == 2:
            probs = probs[0]
        
        # Sample action
        action = np.random.choice(len(probs), p=probs)
        
        log_probs = self.actor.log_softmax(logits)
        if len(log_probs.shape) == 2:
            log_probs = log_probs[0]
        log_prob = log_probs[action]
        
        if isinstance(value, np.ndarray):
            value = value.item()
        
        return action, value, log_prob

    def evaluate_actions(self, states, actions, store_activations=True):
        """
        Evaluate actions during training update.
        Returns values, log_probs, entropy.
        IMPORTANT: Uses single forward pass and respects store_activations for backprop.
        """
        logits, values, features = self.forward(states, store_activations=store_activations)
        
        # Compute log probs for taken actions
        all_log_probs = self.actor.log_softmax(logits)
        batch_size = states.shape[0]
        log_probs = all_log_probs[np.arange(batch_size), actions]
        
        # Compute entropy
        probs = self.actor.softmax(logits)
        entropy = -np.sum(probs * all_log_probs, axis=-1).mean()
        
        return values, log_probs, entropy, logits