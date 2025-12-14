import numpy as np


class PPO:
    def __init__(self, network, lr, clip_epsilon, value_coef, entropy_coef, max_grad_norm):
        """
        PPO Algorithm implementation.
        
        :param network: The ActorCritic network to train
        :param lr: Learning rate (e.g., 3e-4)
        :param clip_epsilon: PPO clipping range (e.g., 0.2)
        :param value_coef: Weight for value loss (e.g., 0.5)
        :param entropy_coef: Weight for entropy bonus (e.g., 0.01)
        :param max_grad_norm: Gradient clipping threshold (e.g., 0.5)
        """
        self.network = network
        self.lr = lr
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def calculate_policy_loss(self, advantages, log_probs, old_log_probs):
        """Calculate clipped surrogate objective"""
        ratio = np.exp(log_probs - old_log_probs)
        clipped_ratio = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        surrogate_1 = ratio * advantages
        surrogate_2 = clipped_ratio * advantages
        policy_loss = -np.mean(np.minimum(surrogate_1, surrogate_2))
        
        # Diagnostic info
        clip_fraction = np.mean(np.abs(ratio - 1) > self.clip_epsilon)
        return policy_loss, ratio, clip_fraction
    
    def calculate_value_loss(self, values, returns):
        """MSE loss for value function"""
        values = values.flatten()
        return np.mean((values - returns) ** 2)

    def compute_policy_gradient(self, advantages, log_probs, old_log_probs):
        """
        Compute gradient of policy loss w.r.t log_probs.
        
        Policy loss = -E[min(r*A, clip(r)*A)]
        where r = exp(log_pi - log_pi_old)
        
        d(loss)/d(log_prob) depends on which surrogate is active
        """
        ratio = np.exp(log_probs - old_log_probs)
        clipped_ratio = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        
        # Gradient only flows through unclipped path
        # When surr1 <= surr2: use surr1, grad = -adv * ratio (since d(ratio)/d(log_prob) = ratio)
        # When surr1 > surr2: use surr2, grad = 0 (clipped, no gradient w.r.t current policy)
        
        dL_dlog_probs = np.zeros_like(log_probs)
        use_surr1 = surr1 <= surr2
        
        # Gradient: d(-surr1)/d(log_prob) = -advantages * ratio
        dL_dlog_probs[use_surr1] = -advantages[use_surr1] * ratio[use_surr1]
        
        # Normalize by batch size
        dL_dlog_probs = dL_dlog_probs / len(log_probs)
        
        return dL_dlog_probs
    
    def compute_value_gradient(self, values, returns):
        """Gradient of MSE value loss w.r.t values"""
        values_flat = values.flatten()
        dL_dvalues = 2 * (values_flat - returns) / len(returns)
        return dL_dvalues.reshape(-1, 1)

    def compute_entropy_gradient(self, logits):
        """
        Compute gradient of negative entropy w.r.t logits.
        
        Entropy H = -sum(p * log(p))
        We want gradient of -H (since we're maximizing entropy as bonus)
        
        d(-H)/d(logits) = p * (log(p) + 1) - p * sum(p * (log(p) + 1))
                       = p * (log(p) + 1 - sum(p * (log(p) + 1)))
        
        Simplified: d(-H)/d(logits)_i = p_i * (log(p_i) - H)
        where H = -sum(p * log(p))
        """
        probs = self.network.actor.softmax(logits)
        log_probs = self.network.actor.log_softmax(logits)
        
        # Entropy per sample
        entropy_per_sample = -np.sum(probs * log_probs, axis=-1, keepdims=True)
        
        # Gradient of -entropy w.r.t logits
        # d(-H)/d(logit_i) = p_i * (log(p_i) + 1 + H)
        dL_dlogits = probs * (log_probs + 1 + entropy_per_sample)
        
        # Average over batch
        dL_dlogits = dL_dlogits / logits.shape[0]
        
        return dL_dlogits

    def log_softmax_backward(self, dL_dlog_probs_selected, logits, actions):
        """
        Backward pass for log_softmax with selected actions.
        
        log_softmax(x)_i = x_i - log(sum(exp(x)))
        
        For the selected action j:
        d(log_softmax_j)/d(x_i) = 1{i==j} - softmax(x)_i
        """
        batch_size = logits.shape[0]
        probs = self.network.actor.softmax(logits)
        
        # Initialize gradient
        dL_dlogits = np.zeros_like(logits)
        
        # For each sample, gradient flows from the selected action
        for i in range(batch_size):
            action = actions[i]
            # d(log_p[action])/d(logits) = one_hot(action) - probs
            dL_dlogits[i, :] = -probs[i, :] * dL_dlog_probs_selected[i]
            dL_dlogits[i, action] += dL_dlog_probs_selected[i]
        
        return dL_dlogits

    def update(self, rollout_buffer, n_epochs, batch_size):
        """Perform PPO update on collected rollout data"""
        data = rollout_buffer.get()
        states = data['states']
        actions = data['actions']
        old_log_probs = data['log_probs']
        advantages = data['advantages']
        returns = data['returns']

        # Normalize advantages (important for stable training!)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # DON'T normalize returns - they're our targets
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # REMOVED
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for epoch in range(n_epochs):
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                batch_indices = indices[start:end]
                
                # Get mini-batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Single forward pass with activation caching
                batch_values, batch_log_probs, entropy, logits = \
                    self.network.evaluate_actions(batch_states, batch_actions, store_activations=True)
                
                # Calculate losses
                policy_loss, ratio, clip_frac = self.calculate_policy_loss(
                    batch_advantages, batch_log_probs, batch_old_log_probs
                )
                value_loss = self.calculate_value_loss(batch_values, batch_returns)
                
                # Accumulate for logging
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += entropy
                num_updates += 1
                
                # === BACKWARD PASS ===
                
                # 1. Policy gradient: d(policy_loss)/d(log_probs)
                dL_dlog_probs = self.compute_policy_gradient(
                    batch_advantages, batch_log_probs, batch_old_log_probs
                )
                
                # 2. Convert log_prob gradients to logit gradients
                dL_dlogits_policy = self.log_softmax_backward(dL_dlog_probs, logits, batch_actions)
                
                # 3. Entropy gradient (we want to maximize entropy, so subtract its gradient)
                dL_dlogits_entropy = self.compute_entropy_gradient(logits)
                
                # 4. Combine actor gradients
                dL_dlogits_total = dL_dlogits_policy - self.entropy_coef * dL_dlogits_entropy
                
                # 5. Value gradient
                dL_dvalues = self.compute_value_gradient(batch_values, batch_returns)

                # 6. Backprop through networks
                dL_dfeatures_actor, actor_grads = self.network.actor.backward(dL_dlogits_total)
                dL_dfeatures_critic, critic_grads = self.network.critic.backward(
                    self.value_coef * dL_dvalues
                )
                
                # 7. Combined feature gradient for CNN
                dL_dfeatures_total = dL_dfeatures_actor + dL_dfeatures_critic
                
                # 8. Backprop through CNN
                _, cnn_grads = self.network.cnn.backward(dL_dfeatures_total)

                # 9. Update weights with gradient clipping
                self.network.actor.update_weights(
                    actor_grads, lr=self.lr, max_grad_norm=self.max_grad_norm
                )
                self.network.critic.update_weights(
                    critic_grads, lr=self.lr, max_grad_norm=self.max_grad_norm
                )
                self.network.cnn.update_weights(
                    cnn_grads, lr=self.lr, max_grad_norm=self.max_grad_norm
                )
            
            # End of epoch logging
            avg_policy = total_policy_loss / num_updates
            avg_value = total_value_loss / num_updates
            avg_entropy = total_entropy / num_updates
            print(f"  Epoch {epoch}: Policy={avg_policy:.4f}, Value={avg_value:.4f}, "
                  f"Entropy={avg_entropy:.4f}, ClipFrac={clip_frac:.3f}")
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }