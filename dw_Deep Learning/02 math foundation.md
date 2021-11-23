# notes2

## Matrix Theory

### matrix

We usually use a 2-dimension array to represent a matrix. in C++, it can be defined as m[i][i]

### tensor

Tensor is the expansion of vector.
scalar is 1 deg tensor, vector is 2 deg tensor, matrix is 3 deg tensor. Above is generally called tensor.

### rank(matix)

Regard a matrix as a volume vector. The rank is the number of its maximal linearly independent group.

### inverse(matrix)

M(square) is reversible when rank(M) $=$ n

### generalized inverse matrix

M is not squared. ABA=A

### matrix decompose

#### character value

A is square matrix, $\exists x, \lambda, Ax=\lambda x$ ,$\lambda$ is A's a character. x is A to $\lambda$ character vector.
we have:

$$
tr(A)=\sum_{i=1}^{n}\lambda_i \\
|A|=\Pi_{i=1}^{n}\lambda_i
$$

we can decompose a matrix by its characters. A=U$\sumU^T$
$\sum$is an opposite angles matrix with all characters on it. U is a combination of n unit vector.

$\forall$A,$\exists$ orthogonal matrix U,V, A=U$\sumV^T$

## Probability Theory

### random variable

- continous
- discrete (enumerated )

### common probablity distribution

- Bernoulli distribution
- binominal distribution
- balance distribution
- normal/Guess distribution
- exponential distribution

### Muti-variable probability distribution

- conditional probability
- joint probability : P(X,Y) = (X , Y happens at the same time)
  the connection: $P(Y|X)P(X)=P(X,Y)$
- prior probability (already know some probability, always based on experience)
- posterior probability (get result to verify the probability)
- total probability:
  $set\ a\ division\ of\ {A_i}, P(B)=\sum P(A_i)P(B|A_i) $
- Bayes (caculate posterior probability):
  $P(A_i|B)=\cfrac{P(B|A_i)P(A_i)}{P(B)}=\cfrac{P(B|A_i)P(A_i)}{\sum P(A_i)P(B|A_i)} $

### common statistical magnitude

#### variance

$Var(X)=E{[x-E(x)]^2}=E(x^2)-[E(x)]^2$

#### covariance

$Cov(X,Y)=E{[x-E(x)][y-E(y)]}=E(xy)-E(x)E(y)$

## information theory

### Entropy

reference:[Stanford it](https://ee.stanford.edu/~gray/it.pdf)
In essence, the "entropy" can be viewed as how much useful information a message is expected to contain. It also provides a lower bound for the "size" of an encoding scheme, in the sense that the expected number of bits to be transmitted under some encoding scheme is at least the entropy of the message. An encoding scheme that manages to achieve this lower bound is called lossless.
$H(X)=-\sum_{i=1}^{n}P(x_i)log_2P(x_i)$
the less H(x) is ,the higher purity of X is and the uncertainty is less

### joint entrophy

just as the covariance

### conditional entrophy

### mutual information

### relative entropy

### cross entropy
