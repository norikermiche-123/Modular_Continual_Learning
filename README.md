Modular Continual Learning via Zero-Leakage Reconstruction Routing

This repository contains the official PyTorch simulation codebase for the paper:

"Modular Continual Learning via Zero-Leakage Reconstruction Routing and Autonomous Task Discovery"

This framework provides a scalable, zero-leakage solution to the Stability-Plasticity Dilemma. By replacing global weight updates with dynamically spawned Task-Specific Experts guarded by local, reconstruction-based routers (Variational ELBO and Tight-Bottleneck Autoencoders), the system achieves mathematically guaranteed 0.0% backward interference without relying on privacy-violating experience replay buffers.

🚀 Key Architectural Features

Zero-Leakage (GDPR Compliant): No historical data buffers or generative replay are used. Data is purged immediately after a localized training session.

The Simultaneous Pipeline: Teacher learning (Plasticity), Student distillation (Stability), and Router manifold acquisition (Task Discovery) are synchronized and optimized in parallel.

Tight-Bottleneck AE ($k=12$): Solves latent space posterior collapse in high-dimensional (4096-D) LLM embeddings, providing a >200x discrimination ratio between crowded semantic tasks.

Autonomous Task Discovery: The system spawns new experts autonomously without human-provided task IDs, utilizing dynamically calibrated MSE thresholds ($\tau_{novelty}$).

🏗️ Architecture Overview

(Place your main PowerPoint 2-panel architecture figure here. Example: docs/architecture_diagram.png)

Panel A: The Simultaneous Pipeline (Training) Raw data passes through a frozen backbone. The resulting latent features $h$ branch into three parallel gradient streams: updating the Persistent Teacher (cross-entropy), distilling into the compact Student Expert (KL divergence), and calibrating the Task Router (MSE/ELBO).

Panel B: Autonomous Routing (Inference) A novel input stream is passed through the library of frozen routers. If recognized by the Familiarity Probe, it is sent to the corresponding expert via Contrastive Soft Routing. If unrecognized, it triggers the spawning of a new module.

📂 Repository Structure

The simulation package is divided into standalone, highly readable scripts designed to replicate specific benchmarks and stress tests from the paper.

Script

Paper Section

Description

simultaneous_split_mnist.py

Section 7.1

Vision baseline. Proves 96.4% retention and $0.0\%$ backward interference via VAE routing.

routing_ablation_study.py

Section 7.2

Tests VAE vs. TB-AE across Raw Pixels, Backbone Latents, and Student Features.

realistic_stress_test.py

Section 7.3

High-dimensional stress test. Simulates highly crowded 4096-D LLaMA-3 embeddings.

scaling_task_retrieval.py

Section 7.4

Simulates lifelong routing: Train A $\to$ Train B $\to$ Return to A. Proves high retrieval SNR.

unified_cl_framework.py

Section 4 & 5

The complete ModularBrain class integrating all sub-components into a single deployable manager.

⚙️ Installation & Requirements

The simulations are lightweight and designed to be run on standard hardware (A CUDA-capable GPU is recommended for speed but not strictly required).

git clone [https://github.com/anonymous-repo/modular-cl.git](https://github.com/anonymous-repo/modular-cl.git)
cd modular-cl
pip install -r requirements.txt


Core Dependencies:

torch >= 2.0.0

torchvision

numpy

📊 Reproducing Paper Benchmarks

1. Vision Baseline (Split-MNIST Retention)

To verify the near-zero fidelity gap (Teacher vs. Student) and the zero-interference constraint via simultaneous distillation:

python simultaneous_split_mnist.py


Expected Result: Student Expert accuracy ~96.40%, Fidelity Gap < 1.0%.

2. High-Dimensional LLM Routing (4096-D)

To verify that the $k=12$ TB-AE bottleneck effectively shatters on out-of-distribution manifolds (simulating dense LLM tasks separated by only 0.8 units):

python realistic_stress_test.py


Expected Result: Discrimination Ratio > 150x between familiar and novel task embeddings.

3. Autonomous Lifelong Retrieval

To simulate the full autonomous lifecycle and verify the Familiarity Probe:

python scaling_task_retrieval.py


Expected Result: The system autonomously recognizes the returning manifold and routes to the existing expert, preventing redundant instantiation.

📜 Citation

(Citation information will be updated upon acceptance and de-anonymization of the double-blind review process).