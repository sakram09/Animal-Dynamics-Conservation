# Animal-Dynamics-Conservation***
**
Automated Wildlife Species Health Assessment using Transfer Learning** 

A deep learning and computer vision framework engineered to process wildlife monitoring data (such as camera trap images) to classify animals and assess their physical health condition. This dry-lab project utilizes MobileNetV2 via Transfer Learning and incorporates real-time image augmentation for high-performance edge deployment in non-invasive conservation monitoring. 

🔬 Problem Statement & Context
Monitoring wildlife health and population dynamics in the wild is critical for conservation biology and preventing biodiversity loss. Traditional field methods require physical trapping or intensive manual inspection of thousands of motion-triggered camera trap images—both of which are slow, labor-intensive, and stressful for the animals.

While automated computer vision can classify species, assessing the health or physical state (e.g., healthy vs. injured, malnourished, or diseased) presents a highly complex spatial pattern-recognition problem. Furthermore, wildlife images collected from the field often suffer from poor lighting, varied camera angles, and limited sample sizes, leading to extreme overfitting in standard deep learning models.

💡 The Computer Vision Solution
This pipeline creates an automated, non-invasive screening tool capable of processing raw wildlife imagery and flags animals showing potential physical anomalies.

Transfer Learning Core: Leverages a pre-trained MobileNetV2 architecture (trained on ImageNet). Using a pre-trained network allows the model to utilize robust low-level features (edges, textures) without needing a dataset of millions of wildlife images.

Edge-Optimized Architecture: MobileNetV2 relies on depthwise separable convolutions, making the final model lightweight, computationally efficient, and perfectly structured for deployment on low-power field edge devices (e.g., smart camera traps).

Robust Image Augmentation: Integrates real-time geometric and pixel-level modifications (rotation_range, shift_range, horizontal_flip) via ImageDataGenerator to simulate dynamic field variables and eliminate model overfitting.

📊 Pipeline Architecture & Implementation Synthetic Population Simulation: 

Generates a tensor array of 200 high-resolution wildlife representation images ($224 \times 224 \times 3$) mapped against binary physical states (Healthy vs. Injured).Feature Extraction Layer: Freezes the foundational convolutional base of MobileNetV2 to preserve universal feature maps.Custom Classification Head: Appends a Global Average Pooling 2D layer, a Dense layer (128 units with ReLU activation), a 40% Dropout layer for regularizing variance, and a Sigmoid output neuron for accurate binary classification.Serialization: Exports the complete architecture, computational graph, and trained weights into a singular optimized Keras model file (.h5).

🚀 Future Research & Scalability Blueprint
This repository serves as a functional baseline that can scale into heavy computational biology or environmental research workflows:

Public Dataset Integration: Replace the synthetic data engine with massive publicly available ecological networks like Snapshot Serengeti or LILA BC (Labeled Information Library for Amazon) datasets.

Multi-Class Disease Phenotyping: Expand the custom dense classification head from binary outputs to categorical paths to classify specific diseases (e.g., Sarcoptic Mange, severe lacerations, or physical starvation markers).

Object Detection Pipelines (YOLO): Transition the architecture into a real-time object detection model like YOLOv8, allowing the system to first locate the animal coordinates in dense forest environments before passing the crop boundaries to the health assessment network. 

