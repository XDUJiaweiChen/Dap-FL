# Dap-FL
Dap-FL: Federated Learning flourishes by adaptive tuning and secure aggregation
Implemented based on Tensorflow

## Reference
- primal-dual ddpg from [Accelerated Primal-Dual Policy Optimization for Safe Reinforcement Learning](https://arxiv.org/abs/1802.06480)

## Contents

### dap_ddpg.py 
define the basic ddpg model with an actor, a critic and a cost

### ddpg_enhanced_second_train.py
execute FL training in dap

### baseline_second_train.py
FL with fixed hyper-paraneters, i.e., FedAvg

### mia.py
model inversation accacks

## Experimental Settings

### Machine Learning Tasks

- Logistic on [MNIST](https://yann.lecun.com/exdb/mnist/)

- CNN on [MNIST](https://yann.lecun.com/exdb/mnist/)

- ResNet on [MNIST](https://yann.lecun.com/exdb/mnist/)

- CNN on [Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

- CNN on [FEMNIST](https://leaf.cmu.edu/)

### Adaptive FL Settings

- *Dap-FL*. (defined in ddpg_enhanced_second_train.py)

- *Large*. (defined in baseline_second_train.py)

- *Small*. (defined in baseline_second_train.py)

- *DDPG-*$\eta$. (defined in ddpg_enhanced_second_train.py by setting fixed training epoch)

- *DDPG-*$\alpha$. (defined in ddpg_enhanced_second_train.py by setting fixed learning rate)

- *DDPG-client*. (see https://ieeexplore.ieee.org/abstract/document/9372789/)

- *DQN*. (see https://ieeexplore.ieee.org/abstract/document/9244624/)
