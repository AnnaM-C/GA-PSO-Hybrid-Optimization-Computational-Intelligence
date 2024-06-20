# Optimizing CIFAR-10 Classification with GA and PSO

## Project Overview
Our project focuses on enhancing neural network optimization using an innovative hybrid algorithm combining Genetic Algorithms (GA) and Particle Swarm Optimization (PSO).
[Comparing Population-Based Algorithms to Fine-Tune a CNN for Image Classification Report](https://github.com/AnnaM-C/computational-intelligence/blob/main/Comparing%20Population-Based%20Algorithms%20to%20Fine-Tune%20a%20CNN%20for%20Image%20Classification.pdf)

## Contribution Highlights
- SSPSO Development: We created the Social Snake Particle Swarm Optimization (SSPSO) algorithm. SSPSO addresses premature convergence in standard PSO by introducing a new mutation probability calculation. Particles closer to the best particle have a higher mutation probability, enabling escape from local minima.
- Comparative Analysis: We evaluated SSPSO against Stochastic Gradient Descent (SGD) and SL-PSO. Despite SSPSO's improvements, SGD outperformed both evolutionary algorithms in neural network optimization.

## Architecture
- VGG-based CNN: We used a simplified VGG-16 architecture with multiple 3×3 convolutional layers and pooling, incorporating dropouts in fully connected layers to improve generalization. The VGG architecture was chosen for its balance between performance and resource efficiency.
  
## Algorithm Justification
- PSO Advantages: PSO requires fewer computational resources and hyperparameter tuning compared to GA. PSOs also converge faster towards global optima.
- SSPSO Enhancement: SSPSO extends SL-PSO by adding a GA-inspired mutation operator, promoting both exploration and exploitation. Particles closer to the best solution are more likely to mutate, aiding escape from local minima.

## Experimental Results
- Setup: We evaluated each algorithm on the CIFAR-10 dataset using a VGG-based CNN, focusing on optimizing the final layer. Experiments were conducted on a computer with an Intel Core i7-8700 CPU, 32GB RAM, and an Nvidia Quadro P4000 GPU.
- Performance: SGD achieved 74% training accuracy and 98% peak accuracy in 100 epochs. SSPSO, while innovative, reached 62% training accuracy and 56% testing accuracy by generation 100, outperforming SL-PSO but not SGD.

## Conclusion
Our SSPSO algorithm provides a novel approach to overcoming PSO's premature convergence issue by integrating GA's mutation mechanism. While it shows promise, further optimization is needed to surpass traditional gradient-based methods like SGD.
