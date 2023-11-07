# siamese networks for duplicate or near-duplicate image detection

**Dataset**: https://www.kaggle.com/datasets/barelydedicated/airbnb-duplicate-image-detection

Siamese Neural Network (SNN) is a neural network architecture containing two or more identical subnetworks. “Identical” here means, they have the same configuration with the same parameters and weights. The update of parameters is reflected on both its subnets simultaneously.


SNN is used to find the similarity of input data by comparing their feature vectors. Some popular applications of SNN include: Face Verification, Signature Verification, Image Seaching System, ...

![image](https://github.com/nttrung2406/siamese-networks-for-duplicate-near-duplicate-image-detection/assets/105348335/e5719fb5-10b5-4edf-bacb-66ab9221f9e0)

**The working flow of SNN**

Select a pair of Input Data (in the scope of this article, images) selected from the dataset.

Pass each image through each Sub-network of SNN for processing. The output of the Sub-networks is an Embedding vector.

Calculate the Euclidean distance between those 2 Embedding vectors.

A Sigmoid Function can be applied on the distance to give a Score value in the interval [0,1], representing the degree of similarity between two Embedding vectors. The closer the score is to 1, the more similar the two vectors are and vice versa.

**Loss function**

a) Triple Loss function

The idea of Triple Loss is to use a set of 3 Input Data including: Anchor (A), Positive (P) and Negative (N) in which the distance from A to P is minimized, while the distance from A is minimized. to N is maximized during model training.

L(A,P,N) = max(||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + \alpha,0)

b) Contrastive loss

The idea of Contrastive Loss is similar to Triplet Loss, the difference is that Contrastive Loss only uses 1 pair of Input Data, either the same type, or different types. If they are of the same type, the distance between their feature vectors will be minimized, and if they are different types, the distance between their feature vectors will be maximized during the training process.

(1 - Y)\frac{1}{2}(D_w)^2 + (Y)\frac{1}{2}{max(0,m - D_w)}^2

D_w is Euclidean distance

\sqrt{{G_w(X_1) - G_w(X_2)}^2}



