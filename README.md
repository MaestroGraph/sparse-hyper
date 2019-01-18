# Sparse, adaptive hyperlayers

This is the codebase that accompanies the paper [Learning sparse transformations through brackpropagation](http://www.peterbloem.nl/publications/learning-sparse-transformations). Follow the link for the paper and an annotated slidedeck.
 
## Disclaimer

We are still cleaning up the code, but it should now be relatively readable. Make sure 
 you have PyTorch 1.0 installed and start by running ```experiments/identity.py```, 
 which runs the identity experiment:
```
 python experiments/identity.py -F
```
The ```-F``` flag sets all values of the matrix to 1, which makes learning a little easier.  
 
Feel free to ask me for help by making an issue, or sending [an email](mailto:sparse@peterbloem.nl).

The ```archive``` branch contains a snapshot of the code at the time the preprint went up.

## Dependencies (probably incomplete)

* Numpy
* Matplotlib
* Pytorch 0.4
* torchvision
* tensorboardX
