
Using python 3.8, install the required libraries with `pip install -r requirements.txt`

To train a network first download the MNIST rotated dataset
Rotated dataset can be found at http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip

Place this data uncompresssed in data/raw

Train a network with:
```python -m train.train_mnist```   
Line 125 of train/train_mnist.py can be changed to train a different model   
This script will save the best method in /saved_models

To test the accuracy of all methods use:
```python -m test.test```