"""pfl — personalised federated learning under differential privacy (T16 / P20).

Modules
-------
config   : default hyperparameters and YAML loading
data     : CIFAR-10 / UCI Adult loading, Dirichlet partition, per-client splits
models   : small GroupNorm CNN and MLP, flat-parameter helpers
dp       : per-example gradients (Opacus or torch.func), clipping + Gaussian noise, RDP accounting
methods  : FedAvg (+ fine-tuning branch), Local-only, Per-FedAvg, IFCA-style clustering, FedPer, q-FFL
metrics  : accuracy, fairness statistics, membership inference, group-fairness gaps
engine   : one end-to-end experiment -> result dictionary
"""

__version__ = "1.0.0"
