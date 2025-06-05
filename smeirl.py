## Author: Mukund Mitra (email: mukundmitra@iisc.ac.in)
## Citation: [1] Mukund Mitra, Gyanig Kumar, Partha Pratim Chakrabarti, and Pradipta Biswas. 2025. Investigating Inverse Reinforcement Learning during Rapid Aiming Movement in Extended Reality and Human-Robot Interaction. J. Hum.-Robot Interact. Just Accepted (May 2025). https://doi.org/10.1145/3736423
## Citation: [2] Mukund Mitra, Preetam Pati, Vinay Krishna Sharma, Subin Raj, Partha Pratim Chakrabarti, and Pradipta Biswas. 2023. Comparison of Target Prediction in VR and MR using Inverse Reinforcement Learning. In Companion Proceedings of the 28th International Conference on Intelligent User Interfaces (IUI '23 Companion). Association for Computing Machinery, New York, NY, USA, 55–58. https://doi.org/10.1145/3581754.3584130
import numpy as np

class SMEIRL:
    def __init__(self, features, learning_rate=0.05, iterations=100):
        self.features = features
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = np.random.randn(features.feature_dim())

    def reward(self, traj):
        return sum(np.dot(self.weights, self.features.compute(state)) for state in traj)

    def log_likelihood_gradient(self, expert_trajs, sample_trajs):
        grad = np.zeros_like(self.weights)
        for tau in expert_trajs:
            grad += self.features.sum_features(tau)
        Z = sum(np.exp(self.reward(tau)) for tau in sample_trajs)
        for tau in sample_trajs:
            prob = np.exp(self.reward(tau)) / Z
            grad -= prob * self.features.sum_features(tau)
        return grad / len(expert_trajs)

    def train(self, expert_trajs, sample_trajs):
        losses = []
        for _ in range(self.iterations):
            grad = self.log_likelihood_gradient(expert_trajs, sample_trajs)
            self.weights += self.lr * grad
            loss = -np.mean([self.reward(tau) for tau in expert_trajs]) + \
                   np.log(np.sum(np.exp([self.reward(tau) for tau in sample_trajs])))
            losses.append(loss)
        return self.weights, losses
