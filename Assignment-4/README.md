Problem 1
------------

* Real labels

![Real labels](plots/toy_dataset_real_labels.png)

* Predicted labels

![Predicted labels](plots/toy_dataset_predicted_labels.png)

* Baboon after compression

![Baboon](plots/compressed_baboon.png)

* GMMs

Random sampling to initialize does not seem to work, rank of many covariance matrices << their size, (48 vs 64 etc), and gets stuck adding the identity matrix * 0.001.

Initializing with k-means seems to work, but it is a much better start than random. I have verified the initial mean and covariance matrices for random sampling, they seem possible.

```
Random Sampling:
MEANS =  [[ 0.37454012  0.95071431]
 [ 0.73199394  0.59865848]
 [ 0.15601864  0.15599452]
 [ 0.05808361  0.86617615]]
COV =  [[ 1.  0.]
 [ 0.  1.]]
PRIOR =  0.25
COV =  [[ 1.  0.]
 [ 0.  1.]]
PRIOR =  0.25
COV =  [[ 1.  0.]
 [ 0.  1.]]
PRIOR =  0.25
COV =  [[ 1.  0.]
 [ 0.  1.]]
PRIOR =  0.25
```

I have tried double checking equations but can't find the bug. (I am using the VM)

![GMM k-means](plots/gmm_toy_dataset_k_means.png)

* GMMs random sampling

![GMM Random](plots/gmm_toy_dataset_random.png)