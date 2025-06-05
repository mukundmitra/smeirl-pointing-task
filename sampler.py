## Author: Mukund Mitra (email: mukundmitra@iisc.ac.in)
## Citation: [1] Mukund Mitra, Gyanig Kumar, Partha Pratim Chakrabarti, and Pradipta Biswas. 2025. Investigating Inverse Reinforcement Learning during Rapid Aiming Movement in Extended Reality and Human-Robot Interaction. J. Hum.-Robot Interact. Just Accepted (May 2025). https://doi.org/10.1145/3736423
## Citation: [2] Mukund Mitra, Preetam Pati, Vinay Krishna Sharma, Subin Raj, Partha Pratim Chakrabarti, and Pradipta Biswas. 2023. Comparison of Target Prediction in VR and MR using Inverse Reinforcement Learning. In Companion Proceedings of the 28th International Conference on Intelligent User Interfaces (IUI '23 Companion). Association for Computing Machinery, New York, NY, USA, 55–58. https://doi.org/10.1145/3581754.3584130
import numpy as np

def generate_quadratic_velocity_profile(start, goal, T=20, a=0.1, b=-0.2, c=0.3):
    """
    Generates a trajectory between `start` and `goal` using a quadratic hand velocity profile.
    """
    direction = goal - start
    distance = np.linalg.norm(direction)
    direction_unit = direction / distance

    times = np.linspace(0, 1, T)
    velocities = a * (1 - times)**2 + b * (1 - times) + c  # v(d) = a*d^2 + b*d + c where d ∝ (1 - t)

    # Normalize velocities to sum to 1 (scaled to total distance)
    velocities /= np.sum(velocities)
    displacements = velocities * distance

    positions = [start]
    for d in displacements:
        positions.append(positions[-1] + d * direction_unit)
    positions = np.array(positions[1:])  # discard start

    traj = []
    for i in range(len(positions)):
        pos = positions[i]
        vel = np.gradient(positions, axis=0)[i]
        acc = np.gradient(np.gradient(positions, axis=0), axis=0)[i]
        jerk = np.gradient(np.gradient(np.gradient(positions, axis=0), axis=0), axis=0)[i]
        traj.append((pos, vel, acc, jerk))

    return traj

def load_demo_data(path='data/pointing/'):
    start = np.array([0, 0, 0])
    goal = np.array([1, 1, 1])
    expert_trajs = [generate_quadratic_velocity_profile(start, goal) for _ in range(10)]
    sample_trajs = [generate_quadratic_velocity_profile(np.random.rand(3), goal) for _ in range(100)]
    return expert_trajs, sample_trajs
