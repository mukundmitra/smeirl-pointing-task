## Author: Mukund Mitra (email: mukundmitra@iisc.ac.in)
## Citation: [1] Mukund Mitra, Gyanig Kumar, Partha Pratim Chakrabarti, and Pradipta Biswas. 2025. Investigating Inverse Reinforcement Learning during Rapid Aiming Movement in Extended Reality and Human-Robot Interaction. J. Hum.-Robot Interact. Just Accepted (May 2025). https://doi.org/10.1145/3736423
## Citation: [2] Mukund Mitra, Preetam Pati, Vinay Krishna Sharma, Subin Raj, Partha Pratim Chakrabarti, and Pradipta Biswas. 2023. Comparison of Target Prediction in VR and MR using Inverse Reinforcement Learning. In Companion Proceedings of the 28th International Conference on Intelligent User Interfaces (IUI '23 Companion). Association for Computing Machinery, New York, NY, USA, 55–58. https://doi.org/10.1145/3581754.3584130
import numpy as np

def save_model(weights, path='trained_weights.npy'):
    np.save(path, weights)

def load_model(path='trained_weights.npy'):
    return np.load(path)
