In [this project](https://github.com/mbastola/neural-nets-in-python/tree/master/deep-neural-nets/ann-tensorflow), we build ANNs using tensorflow library as an extension to ann-numpy repo where I build ANN architecture from scratch in numpy. 

## MNIST:

![png](https://github.com/mbastola/neural-nets-in-python/blob/master/deep-neural-nets/ann-numpy/imgs/mnist.png)

Our tensorflow ANN with 1 hidden layer with only 30 nodes on 10 classes handwritten digit MNIST dataset achieves test error of 3.4% (accuracy 96.6% !!). 

![png](https://github.com/mbastola/neural-nets-in-python/blob/master/deep-neural-nets/ann-tensorflow/imgs/ANN_mnist10.png)

**training_error_rate**

final training error:  0.033

test error: 0.03428571428571429

Confusion Matrix:
```
[[67  0  0  0  0  0  2  0  0  0]
 [ 0 77  1  0  0  0  0  0  0  0]
 [ 0  0 62  0  2  0  0  0  0  0]
 [ 0  0  1 86  0  1  0  0  0  0]
 [ 0  0  0  0 62  0  1  0  0  1]
 [ 0  0  0  3  0 60  0  0  0  0]
 [ 0  0  0  0  0  1 66  0  0  0]
 [ 0  0  0  0  0  0  0 67  0  0]
 [ 2  0  0  0  1  3  1  0 63  0]
 [ 0  0  0  0  0  1  0  1  2 66]]
```

Classification Report:

```
              precision    recall  f1-score   support

         0.0       0.97      0.97      0.97        69
         1.0       1.00      0.99      0.99        78
         2.0       0.97      0.97      0.97        64
         3.0       0.97      0.98      0.97        88
         4.0       0.95      0.97      0.96        64
         5.0       0.91      0.95      0.93        63
         6.0       0.94      0.99      0.96        67
         7.0       0.99      1.00      0.99        67
         8.0       0.97      0.90      0.93        70
         9.0       0.99      0.94      0.96        70

    accuracy                           0.97       700
   macro avg       0.97      0.97      0.96       700
weighted avg       0.97      0.97      0.97       700

```            

## CIFAR 10

![png](https://github.com/mbastola/neural-nets-in-python/blob/master/deep-neural-nets/ann-numpy/imgs/cifar10.png)

For 10 class 3 channel images CIFAR dataset we try deeper archecture with 3 hidden layers of sizes 128, 64 and 32 nodes. The test error is 61.3% which is a bit worse than using my numpy version with test error of 53.1%. I've used CNNs to get down to 18% test error in my CNN folders. 


![png](https://github.com/mbastola/neural-nets-in-python/blob/master/deep-neural-nets/ann-tensorflow/imgs/ANN_cifar10.png)

**training_error_rate**


final training error: 0.647

test error: 0.613333333333

Confusion Matrix:

```
[[47  3  2  3  0  4  1  3  0  2]
 [10 38  2  3  0  1  2  1  0 10]
 [ 9  4 16  5  7  5  8  3  0  0]
 [ 3  4  6 14  0 22  4  2  0  1]
 [12  1  8  2 19  5  7  5  0  0]
 [ 0  3  1  6  5 17  7  2  0  1]
 [ 1  1  6  6 15  1 29  4  0  2]
 [ 7  3  4  3 13  3  1 32  0  2]
 [31  5  2  3  1 10  0  2  0 15]
 [ 7 12  2  2  2  1  2  4  0 20]]
```
Classification Report:

```
              precision    recall  f1-score   support

         0.0       0.37      0.72      0.49        65
         1.0       0.51      0.57      0.54        67
         2.0       0.33      0.28      0.30        57
         3.0       0.30      0.25      0.27        56
         4.0       0.31      0.32      0.31        59
         5.0       0.25      0.40      0.31        42
         6.0       0.48      0.45      0.46        65
         7.0       0.55      0.47      0.51        68
         8.0       0.00      0.00      0.00        69
         9.0       0.38      0.38      0.38        52

    accuracy                           0.39       600
   macro avg       0.35      0.38      0.36       600
weighted avg       0.35      0.39      0.36       600
```
