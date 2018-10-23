# Sparse, adaptive hyperlayers

This is the codebase that accompanies the paper [Learning sparse transformations through brackpropagation](http://www.peterbloem.nl/publications/learning-sparse-transformations). Follow the link for the paper and an annotated slidedeck.
 
## Disclaimer

The code is currently a mess. If you're willing to get your hands dirty, make sure you have PyTorch 0.4 installed and start by running ```identity.experiment.py```, which runs the identity experiment:
```
 python identity.experiment.py -F -G
```
 
Feel free to ask me for help by making an issue, or sending [an email](mailto:sparse@peterbloem.nl).

I'll start cleaning up soon. The archive branch contains a snapshot of the code at the time the preprint went up (in case you get started and something suddenly disappears).

## Dependencies (probably incomplete)

* Numpy
* Matplotlib
* Pytorch 0.4
* torchvision
* tensorboardX
