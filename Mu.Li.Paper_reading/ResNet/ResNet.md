# ResNet

Deep Residual Learning for Image Recognition

非常深的 NN 难以训练，用一个残差的网络替换别人的 CNN

# notes1:

## category of AI

- weak AI
- strong AI
- supervise AI

## Machine Learning

### category

- supervised ,eg: Naive Bayes / SVM / Linear, Logestic Regression
- unsupervised ,eg: Feature Selection / K-Means
- reinforcement: UC Berkeley EECS CS418 "Paceman".

### classic algorithm

#### K-Means (From Stanford University)

K-means clustering is a simple and elegant approach for partitioning a data set into K distinct, nonoverlapping clusters. To perform K-means clustering, we must first specify the desired number of clusters K; then, the K-means algorithm will assign each observation to exactly one of the K clusters.
[what is clustering?]
Clustering is the task of dividing the population or data points into several groups such that data points in the same groups are more similar to other data points in the same group than those in other groups.
The core idea of K-Means algorithm is as belows:
If we have $n$ observations and we want to divide them into K groups, that's why we call it as K-Means! We also call each group as cluster. And we denote the action of dividing as clustering. What we want most is that we want to minimize the difference between the observations within a single cluster. So, how to quantify the difference is a problem. Usually we use **squared Euclidean distance** to measure it.
The standardized squared Euclidean distance is defined as belows:
![1](截屏2021-11-17%2013.45.15.png)
In other words, the within-cluster variation for the kth cluster is the sum of all of the pairwise squared Euclidean distances between the observations in the kth cluster, divided by the total number of observations in the kth cluster. Combining all the equations gives the optimization problem that defines K-means clustering.
How to find optimal K is also the point the K-means do! We usually use Elbow Method andsilhouette method.

The working of the K-Means algorithm is as blows:

- randomize a K
- randomly select K values and assign them in a single cluster
- random select a centroid of each cluster
- compute the squared Euclidean distances between each nodes and centroids, and assign the closest to the centroid.
- Recompute the centroids of newly formed clusters
- Repeat steps 3 and 4 until the centroids never change!

reference: [mlweb](https://towardsmachinelearning.org/k-means/)
[Stanford](https://nlp.stanford.edu/IR-book/html/htmledition/pivoted-normalized-document-length-1.html#p:euclideandistance)
[Stanford slides](http://www.econ.upf.edu/~michael/stanford/maeb4.pdf#:~:text=The%20squared%20Euclidean%20distance%20sums%20the%20squared%20differences,gives%20exactly%20the%20same%20value%200.6%20as%20before.)

### famous research institution and scholars

#### [Geoffrey Hinton](https://www.cs.toronto.edu/~hinton/)

#### [Deepmind at Google](https://deepmind.com/)
