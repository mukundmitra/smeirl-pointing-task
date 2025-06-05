## Author: Mukund Mitra (email: mukundmitra@iisc.ac.in)
## Citation: [1] Mukund Mitra, Gyanig Kumar, Partha Pratim Chakrabarti, and Pradipta Biswas. 2025. Investigating Inverse Reinforcement Learning during Rapid Aiming Movement in Extended Reality and Human-Robot Interaction. J. Hum.-Robot Interact. Just Accepted (May 2025). https://doi.org/10.1145/3736423
## Citation: [2] Mukund Mitra, Preetam Pati, Vinay Krishna Sharma, Subin Raj, Partha Pratim Chakrabarti, and Pradipta Biswas. 2023. Comparison of Target Prediction in VR and MR using Inverse Reinforcement Learning. In Companion Proceedings of the 28th International Conference on Intelligent User Interfaces (IUI '23 Companion). Association for Computing Machinery, New York, NY, USA, 55–58. https://doi.org/10.1145/3581754.3584130
from smeirl import SMEIRL
from features import FeatureExtractor
from sampler import load_demo_data
from utils import save_model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    expert_trajs, sample_trajs = load_demo_data()
    fe = FeatureExtractor()
    irl = SMEIRL(features=fe, learning_rate=0.1, iterations=50)
    weights, losses = irl.train(expert_trajs, sample_trajs)

    print("Trained weights:", weights)
    save_model(weights)

    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('SMEIRL Training Loss')
    plt.grid(True)
    plt.show()
