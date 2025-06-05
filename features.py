## Author: Mukund Mitra (email: mukundmitra@iisc.ac.in)
## Citation: [1] Mukund Mitra, Gyanig Kumar, Partha Pratim Chakrabarti, and Pradipta Biswas. 2025. Investigating Inverse Reinforcement Learning during Rapid Aiming Movement in Extended Reality and Human-Robot Interaction. J. Hum.-Robot Interact. Just Accepted (May 2025). https://doi.org/10.1145/3736423
## Citation: [2] Mukund Mitra, Preetam Pati, Vinay Krishna Sharma, Subin Raj, Partha Pratim Chakrabarti, and Pradipta Biswas. 2023. Comparison of Target Prediction in VR and MR using Inverse Reinforcement Learning. In Companion Proceedings of the 28th International Conference on Intelligent User Interfaces (IUI '23 Companion). Association for Computing Machinery, New York, NY, USA, 55–58. https://doi.org/10.1145/3581754.3584130
import numpy as np

class FeatureExtractor:
    def __init__(self, max_vals=np.array([1, 1, 10, 10])):
        self.max_vals = max_vals

    def compute(self, state):
        pos, vel, acc, jerk = state
        fd = np.exp(-np.linalg.norm(pos))  # Distance-based
        v_des = 0.1 * np.linalg.norm(pos)**2 - 0.2 * np.linalg.norm(pos) + 0.3  # Quadratic desired velocity
        fv = -np.square(np.linalg.norm(vel) - v_des)  # Velocity deviation
        fa = np.sum(np.square(acc))  # Acceleration penalty
        fj = np.sum(np.square(jerk))  # Jerk penalty
        return np.array([fd, fv, fa, fj]) / self.max_vals

    def feature_dim(self):
        return 4

    def sum_features(self, traj):
        return sum(self.compute(s) for s in traj)

