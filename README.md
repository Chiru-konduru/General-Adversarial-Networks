# General-Adversarial-Networks
## Generative Adversarial Networks (GANs) for Cat Face Generation

This repository is dedicated to the implementation and training of Generative Adversarial Networks (GANs) on a cat dataset with the aim of generating cat face images. This project serves as an exciting journey into the world of generative models, specifically exploring the fascinating concept of GANs and data augmentation. The project is adapted and inspired from Stanford CS231n Assignments, CMU 16-726 (Assignment 3)[https://learning-image-synthesis.github.io/sp21/assignments/hw3] , and PyTorch Tutorials.

## Table of Contents

- [Description](#Description)
- [Implementation](#Implementation)
- [Extra Credit](#Extra-Credit)
- [Acknowledgements](#Acknowledgements)

## Description

The focus of this project is on the development and training of generative models known as GANs. These networks are used to generate realistic cat face images from random noise inputs. The training process involves using a cat dataset and making use of the powerful PyTorch library for implementation.

## Implementation

A ResNet18 architecture was trained on a rotation task, utilizing a CIFAR10 dataloader for implementation. We implemented and trained two different models: the original GAN and LSGAN (Least Squares GAN), which uses a different loss function. The generator and discriminator network architectures are based on the Deep Convolutional GAN (DCGAN). 

Data augmentation was performed using PyTorch's built-in transforms, further enhancing the model's learning process and generation quality.

The top-level notebook, `MP4.ipynb`, provides a detailed step-by-step guide to implementing and training the GAN. 

The `GAN_debugging.ipynb` notebook, which provides a smaller network for training on the MNIST dataset, is also included. This debugging notebook allows quick verification of the implemented loss functions and training code correctness before proceeding with the main assignment.

## Extra Credit

For extra credit, several advanced concepts and experiments were carried out:

1) We experimented with alternative GAN losses such as the margin-based WGAN/WGAN-GP, DRAGAN, and BEGAN.
2) An advanced normalization scheme was implemented - spectral normalization.
3) The GAN was also trained on a different dataset to test its generalization capabilities.

## Acknowledgements

This project was part of a course assignment, and we'd like to express our gratitude to the instructors and TAs for their guidance and support.

Note: This project is for educational purposes and should be used responsibly.

---

Refer to the `MP4.ipnyb` and `GAN_debugging.ipnyb` notebooks for more detailed project information, implementation, results, and discussions. For any questions or suggestions, feel free to open an issue.
