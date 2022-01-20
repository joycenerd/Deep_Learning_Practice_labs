# Deep_Learning_Practice_labs
Labs for 5003 Deep Learning Practice course in summer term 2021 at NYCU.

## Back-propagation 

Understand and implement simple neural networks with forwarding pass and backward propagation using only two hidden layers.
* Linear data accuracy: 99%
* XOR data accuracy: 100%

:page_facing_up: [Report](./lab1/REPORT.pdf) :computer: [Code](./lab1)


## EEG Classification

Implementing two EEG classification model which are EEGNet and DeepConvNet. Also, we are changing the activation function (ELU, ReLU, LeakyReLU) in the model and see the difference.

* EEGNet accuracy: 87.87%
* DeepConvNet accuracy: 76.48%

:page_facing_up: [Report](./lab2/REPORT.pdf) :computer: [Code](./lab2)

## Diabetic Retinopathy Detection

Analyze diabetic retinopathy using ResNet, compare the results of ResNet18, ResNet50, and both networks with pre-trained weights.

* ResNet50 with pretrained accuracy: 82.18%
* ResNet18 with pretrained accuracy: 79.24%
* ResNet50 w/o pretrained accuracy: 73.55%
* ResNet18 w/o pretrained accuracy: 73.35%

:page_facing_up: [Report](./lab3/docs/REPORT.pdf) :computer: [Code](./lab3)

## Conditional sequence-to-sequence VAE

Implementing conditional seq2seq VAE for English tense conversion and generation.

monotonic KL annealing
* BLEU score: 0.8312
* Gaussian score: 0.472

cyclical KL annealing
* BLEU score: 0.9527
* Gaussian score: 0.452

:page_facing_up: [Report](./lab4/REPORT_v2.pdf) :computer: [Code](./lab4)

## Let's Play GANs

Implement conditional GAN to generate synthetic images in multi-label conditions.

* SAGAN acc: 77.22%
* cDCGAN acc: 68.89%
* WGAN acc: 60.56%

:page_facing_up: [Report](./lab5/REPORT.pdf) :computer: [Code](./lab5)

## Deep Q-Network and Deep Deterministic Policy Gradient

Implement DQN and DDQN to solve LunarLander-v2. Implement DDPG to solve LunarLanderContinuous-v2.

* DQN average reward: 277
* DDPG average reward: 273
* DDQN average reward: 256

:page_facing_up: [Report](./lab6/REPORT.pdf) :computer: [Code](./lab6)

