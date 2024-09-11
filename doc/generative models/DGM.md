# Deep Generative Models

Lecture notes of Stanford CS236 2023 Deep Generative Models

## Lecture 1 - Introduction

In general we define generative models as a learnt probability distribution 
$p(x)$ from which we sample our generated data.


## Lecture 2 - Background

We define our data as samples from a data distribution $x \sim P_\text{data}$
Through the means of deep learning techniques we then try to find a good
approximation for $P_\text{data} \approx P_\theta$ 

$\theta \in M$ means that $\theta$ refers to a parameterization of a given model
which means a model family $M$ Once you defined the set $M$ the goal becomes to
find a good approximation which means to optimize a distance function 
$d(P_\text{data}, P_\theta)$ which measures the distance between the two

Similarly to how $x \sim P(x)$ samples new data from our distribution
where $x$ will be somewhat likely, the inverserse $P(x)$ should be high if x
is part of the distribution. This we call **density estimation**

Suppose we model a pixel as a distribution over the possible pixel values 
$P(R,G,B)$ each one of the random variables can have a state ranging from 
$0..255$. Each of these states have to have a probabity assigned to them. In 
addition we need to define the probability for each combination of the 3. So 
for just one pixel this would mean we have to define $256^3$ states. This means
having $16581375$ possible states.

Now if we had a sixteen by sixteen image this would of course be 
$(256^3)^{16^2} = (256^{3*256}) = (256^{768})$ states. So you see its not 
possible to model this. 

If you could assume the pixel values to be independent you could use the rule 
that $p(x_1, ..., x_2) = p(x_1)p(...)p(x_2)$ Which would amount to the same 
amount of states however much less memory requirements for modelling. You cant
however assume pixels of an image to be independant so this doesnt work anyway

We can however make use of 2 important concepts:
Let $S$ be an event
$$ 
\text{Chain Rule} := p(S_1 \cap S_2 \cap ... \cap S_n)
= p(S_1)p(S_2 | S_1)...p(S_n|S_1\cap...\cap S_{n-1})
$$

$$ 
\text{Bayes Rule} := p(S_1|S_2) = \frac{p(S_1 \cap S_2)}{p(S_2)} =
\frac{p(S_2 | S_1)p(S_1)}{p(S_2)}
$$

also keep in mind that $p(A \cap B) = p(A,B)$

S can be a normal random variable

The chain rule still requires an exponential amount of parameters given that
every pixel is dependant on every other pixel. However lets suppose that only
the previous pixel is a condition and all other pixels are **conditionally 
independant**: $x_{n+1} \bot x_1, ..., x_{n-1} | x_n$ This specific conditionally
independant model is in fact a markovian model, where each step only depends
on the previous

In a bayesian network we can generalize this idea to a conditional parameterization
this means that for each random variable $x_i$ we define a set of random
variables $x_{A_i}$ which define the conditions. In this case 
$p(x_1,...x_n) = \prod_i p(x_i|x_{A_i})$ The bayesian network now is just a way
to structure these conditionals in a graph structure. In this case each 
conditional set is defined only by the parents of a node. Each random variable
here is represented by a single node. This structure implies conditional
independances

Now, using the chain rule $p(Y,X) = p(X|Y) p(Y) = p(Y|X)p(X)$
Here the first example would be of a generative nature, the second of a 
discriminatory one.

One attempt to reduce the required effort of the conditional states is to define
conditional independance, another is to define a function that maps a condition
vector to a probability. $p(Y = 1 | x; \alpha) = f(x, \alpha)$ This is what we 
do in logistic regression, where $\alpha$ is a vector of parameters. So 
essentially we substitute the conditional independance of all but one random
variable, which may be too simplistic, by a function that maps all conditional 
random variables to a probabilistic value for the specific y. So we substitute 
the table that has all of these conditions and y with just this function. We do
this in the classifier because we just take the x as given and dont need to care
about the relation that they have to each other, we only care about the relation
between them and the y

So it becomes that $z(x, \alpha) = \alpha_0 + \sum_{i=1}^n\alpha_ix_i$, however
we need to ensure that this sums up to 1 for every value of x so we take 
$p(Y=1|\vec{x};\alpha) = \sigma (z(\vec{x},\alpha))$ where $\sigma$ is the 
sigmoid

So a classifying neural network captures at its very core a conditional probability
of the class given the observations

Applying this to the chain rule earlier we would have something like
$$ 
\text{Chain Rule} := p(S_1 \cap S_2 \cap ... \cap S_n)
= p(S_1)p(S_2 | S_1)...p_\text{neural}(S_n|S_1\cap...\cap S_{n-1})
$$

So far we talked about discrete values for the random variables. In this case
we represent the probability assigned to each random variable using a 
**probability mass function**. If however we decide
to have continuous variables which are assigned variables in the form 
$p_x:\R \rightarrow \R^+$ we speak of a **probability density function**. One
of such functions for example is the Gaussian $X \sim \mathcal N(\mu, \sigma)$
or if X is a vector $X \sim \mathcal N(\mu, \Sigma)$

The chain rule and bayes rule still fully apply to density functions. Now you 
can also mix different density functions together for example take the simple 
2 node bayesian network with random variables $Z, X : Z \rightarrow X$

in this case 
$$
p_{Z,X} (z,x) = p_Z(z)p_{X|Z}(x | z)
$$
and
$$
Z \sim \text{Bernoulli(p)}
$$
$$
X | (Z = 0) \sim \mathcal N (\mu_0, \sigma_0), 
X | (Z = 1) \sim \mathcal N (\mu_1, \sigma_1), 
$$
Which is just mixture of 2 gaussians depending on what value Z takes. 

However you can also look at a different example the **variational autoencoder**
$Z, X : Z \rightarrow X$
$$
p_{Z,X} (z,x) = p_Z(z)p_{X|Z}(x | z)
$$

$$
Z \sim \mathcal N (0, 1)
$$
$$
X | (Z = z) \sim \mathcal N (\mu_\theta(z), e^{\sigma_\phi(z)}) 
$$

Keep in mind that the bayesian model is a generative model that given looks for
$p(Y,X) = p(X|Y) p(Y)$ So we have a label or something and assign a probability
to the individual inputs. This requires a complex modelling of the relation of
all x to the corresponding y

## Lecture 3 - Autoregressive Models

Suppose you want to learn the MNIST dataset consisting of digits in binary 
28x28 images. So you learn to approximate the data distribution 
$p(x_1,..,x_{784})$ over $x\in \{0,1\}^{784}$ such that $x \sim p(x)$ looks like 
a digit

This requires a two step process
- Parameterize a model family $\{p_\theta(x), \theta \in \Theta\}$ 
- Search for the right model parameters $\theta$ based in the data  

### Fully Visible Sigmoid Belief Network

This is basically modelling the predictive task as 
$p(S_1)p(S_2 | S_1)...p_\sigma(S_n|S_1\cap...\cap S_{n-1})$

So you basically take the joint distribution of a whole lot of classification 
models. like weve seen before each conditional is a classification problem.

We call this approach **autoregressive** because it uses each newly generated
data in the next generation step. > To sample you first decide the first value
and then take that as the conditional for the next classifier function etc.

Because the parameters increase linearly with each new conditional the overall
parameter requirements is quadratic. Which is a big improvement over the
exponential from modelling everything. However the results are rather 
lackluster.

### Neural Autoregressive Density Estimation 

Basically what we did before except now its no longer logistic classification
but a neural classifier

$p(S_1)p(S_2 | S_1)...p_\text{neural}(S_n|S_1\cap...\cap S_{n-1})$

Here the weights can also progressively expand meaning that you always take the
same weight vector for the same condition and in each conditional you add one 
of those vectors to the weight matrix

Depending on what you want each prediction step to be, ie every conditional to
evaluate to, you define the output as different distributions. 
Where as in binary classification for each conditional you would see it as a
bernoulli or sigmoid function, or in a multiclass classification a softmax. 
In case of a continuous model it would be a density function and you would have
to predict the parameters of that density function such as a gaussian.

This is logical because we already predicted the parameters for the classifications.
Just that through the discrete nature of those distributions we have as many 
classes as we do probabilities. We then through the means of softmax construct
the probability mass function from each class value or logit for the individual
conditional. 

Similarly we do this for the continuous one, but since a density function is 
not characterized by classes, you have to predict other values. such as the mean
and variance of a gaussian, or multiple per step if its a mixed gaussian.

FVSBN and NADE look similar to an **autoencoder**

with an encoder $e(x) = \sigma(W^2(W^1x+b^1)+b^2)$ and a decoder such that
$d(e(x))\approx x$ so like $d(h) = \sigma(Vh+c)$

Inherently an autoencoder does not keep the autoregressive property that any 
output can only depend on all previous outputs. However using masking you can
achieve this.

We call this **Masked Autoencoder for Distribution Estimation** The advantage
for this over NADE for example is that we can make each prediction in a single
pass using the same neural network, while still sticking to the general principal
of bayesian networks

### Recurrent Neural Network

Another approach to do something like this is to use an RNN to model
$p(x_t|x_{1:t-1};\alpha^t)$ which as you can see has a very similar structure.
with $x_{1:t-1}$ being the history

The main idea however is to not keep the entire history and build each next step
on all conditionals that come before it, but to rather keep a "summary" of those
conditionals and update that as a proxy for all previous conditions 


## Lecture 4 Maximum Likelihood learning

Attention based models are more similar again to NADE in that they have direct
access to all previous timesteps. However due to the masked multi head attention
in transformers, they still preseve the autoregressive structure.

### PixelCNN

At each convolutional step you predict what the current pixel should be. This
requires masking of not yet predicted pixels. Notice how this is done for 
the generation or sampling of new images, NOT the usual structure learning of
CNN. So essentially its sort of like an RNN or again an autoregressive model.

### Summary of Auto Regressive Models

Autoregression is useful because it is easy to sample from these methods. This
is because you only have to look at one conditional at a time.

At the same time it is easy to compute a probabilty.

Works good for continuous and discrete values.

**However** No good way to get features, cluster points or to do unsupervised
learning.

### Learning

Lets assume that the domain is governed by a data distribution $P_\text{data}$
From this distribution we are given a dataset $\mathcal D$ of m samples

Each sample in this dataset is an assignment to a subset of the random variables
of the distribution. 

The standard assumption is that the date instance are independent and 
identically distributed **(IID)**

Further we are given a family of Models $\mathcal M$ and our task is to learn a 
good distribution in this set

The ideal goal is that $P_\theta = P_\text{data}$

However this is in general not achievable because limited data can only provide
an approximation of the underlying data, and also for computational reasons

Now the question becomes what is the "best" distribution. Depending on what we 
want to do this could be multiple different things.
- Density Estimation: interested in the full distribution so we can compute
whatever conditional probabilities (big problem)
- **Structured Prediction**: Use the distribution to make a prediction in a
specific domain
- Structure or knowledge discovery in data

### Learning the full distribution

Minimize $d(P_\text{data}, P_\theta)$ now how do we compute d? - For auto 
regressive models a standard approach is to measure the likelihood. To this end
we use the **KL-Divergence** (Kullback-Leibler)
which measures the distance between two  distributions as follows:
Let $p$ and $q$ be two distributions, then
$$
D_\text{KL}(p||q) = \sum_x p(x)\log \frac{p(x)}{q(x)}
$$
This is basically the expectation with respect to all the possible things that 
can happen, weighted by the log of the ratios of the probabilities assigned by 
p and q respectively. This measure has the properties that its always positive
as well as only being 0 if and only if $p=q$. Through the division and 
multiplication this measure is asymmetric. \
(Measures the expected number of extra bits required to describe samples from
p(x) using a compression code based on q instead of p)

minimizing the KL Divergence is equivalent to maximizing the **log-likelihood**
This is because

$$
D_\text{KL}(P_\text{data}||P_\theta) = P_\text{data}(x)
\log \frac{P_\text{data}(x)}{P_\theta(x)}
$$
$$
= \mathbf{E_{x\sim P_\text{data}}} [\log \frac{P_\text{data}(x)}{P_\theta(x)}]
$$
$$
= \mathbf{E_{x\sim P_\text{data}}}[ \log P_\text{data}(x)] -
\mathbf{E_{x\sim P_\text{data}}} [\log P_\theta(x)]
$$
And the first term does not depend on $P_\theta(x)$ However because we ignore
the first term we cannot know how close we are to the optimum, because we are
no longer computing a distance.

Also reminder, Expectation of a random variable is defined as:
$$
 \mathbf{E_{x\sim P}}(P(x)) := \sum_x x p(x)
$$

This of course still includes the original data distribution $P_\text{data}$
which we dont know. So instead we can approximate the expected log likelihood

$$
\mathbf{E_{x\sim P_\text{data}}}(\log P_\text{data}(x))
$$

with the **empirical log likelihood**

$$
\mathbf{E_{\mathcal D}}(\log P_\theta(x)) = \frac{1}{|\mathcal D|}
\sum_{x \in \mathcal D}\log P_\theta(x)
$$

Equivalently you are also maximizing the likelihood of the distribution
$$
\prod_{x \in \mathcal D}P_\theta(x)
$$

This is easy for an auto regressive model.

$$  
L(\theta, \mathcal D) = \prod_{j=1}^m P_\theta(x^{(j)})
= \prod_{j=1}^m \prod_{i=1}^n P(x_i^{(j)}|x_{1:i-1}^{(j)};\theta_i)
$$

$$  
\nabla_\theta \mathcal l(\theta, \mathcal D) = \prod_{j=1}^m P_\theta(x^{(j)})
= \prod_{j=1}^m \prod_{i=1}^n \nabla_\theta \log  
P_\text{neural}(x_i^{(j)}|x_{1:i-1}^{(j)};\theta_i)
$$

or in case of Monte Carlo approximation using samples (Mini Batches)

$$  
\nabla_\theta \mathcal l(\theta)
= m \mathbf{E_{x^{(j)}\sim \mathcal D}} \prod_{i=1}^n \nabla_\theta \log  
P_\text{neural}(x_i^{(j)}|x_{1:i-1}^{(j)};\theta_i) , m = | \mathcal D|
$$

### Cons of Autoregressive Models

- Requires an ordering of the conditionals
- Generation is sequential based on that order
- No unsupervised learning

## Lecture 5 - VAEs 1

### Latent variable models

Images have a lot of variability, such as different hair colour, pose, etc...
These factors of variation however are not explicitly available unless given as 
an annotation. We call these hidden factors of variation **latent**. \
The Idea then becomes to model these factors using new **latent variables z**

Informally speaking with a bayesian network, we want all z to act as conditionals
for our data x $Z\rightarrow X$

if z is chosen properly $p(x|z)$ can be much simpler than the marginal $p(x)$
also you can identify the latent variables when you have a model $p(z|x)$. The
challenge becomes then to learn these latents

To begin with we simply start by defining a set of latents and assigning them a
simple distribution $\vec{z} \sim \mathcal N(0,I)$ This essentially means that
each latent is just a normally distributed value if we define all conditionals
as a vector.
Then $p(x|z) = \mathcal N(\mu_\theta(\vec{z}),\Sigma_\phi(\vec{z}))$ So again,
like we did with approximating the relation of conditionals to the specific 
variable in the chain rule application for auto regressive models, we do the 
same here. Except now we approximate the relation of th econditional latent 
variables which we simply define as easy gaussians to the data using neural
networks that predict the parameters of the defining distribution $p(x|z)$

Generative Process
- Sample a z 
- Generate data point by sampling from a gaussian

Z, through the continuous nature, represents one of an infinite amount of values.
This sort of acts as the category that then gets transformed by the neural networks
but from this specific z you can generate many gaussians

$$
\text{prior} = \vec z \sim \mathcal N (0, I) 
$$
$$
p(x|z) = \mathcal N(\mu_\theta(z), \Sigma_\phi(z))
$$
$$
\mu_\theta(\vec{z}) = \sigma(\mathbf A\vec{z} + c) \
= (\sigma (\vec a_1 \vec z + c_1), \sigma (\vec a_2 \vec z + c_2))
= (\mu_1(\vec z), \mu_2(\vec z))
$$

This essentially means, that we sample a random vector z each component
represents one latent factor of variation (basically a feature such has 
hair colour). Then based on this specific sample of features we then generate
a gaussian. Since the z influences this, the overall structure is a guassian 
mixture. Each gaussian in the mixture can then learn to prioritize different 
features. For example if one looks for brown hair, and the feature has a low 
chance of that, the gaussian will be different etc.

The last term is the mapping of the latent to each real dimension of the real 
data. So a latent with features brown hair and blue eyes would map to first a 
mean red value for dimension r of the first pixel etc etc. \
Then at the end you do a sample from this normal distribution to get your final 
x

### Marginal likelihood

Challenge is some part of the data is missing. I dont understand why thats a 
real problem, but i believe this is meant to mean that $\mathcal D $ is not
$P_\text{data}$ so we are missing data from that distribution? Ok, no this is
meant to represent that we dont know the zs. because they are a conditional
that we do not know for a training image

Suppose X denotes observed random variables, Z denotes unobserved ones (hidden
or latent)

We have a model for the joint distribution such as the PixelCNN
$p(X,Z; \theta) $

What is the probability $p(X = \bar x; \theta)$ of observing a training data
point $\bar x$? So we are asking what the marginal probability of a specific
observed region is, while not knowing the unobserved region.

$$
\sum_{\vec z} p(X = \bar x, Z = z; \theta) = \sum_{\vec z} p(\bar x, z; \theta)
$$

So you need to sum the probability of all options for z, which is of course
a possibly infinite range \
For a VAE we have the same problem at training time, because we dont know the
latents z for the observed data. So in order to estimate the probability of
x we need to try every possibility for z and sum them up. \
The above example is a discrete one, but in our case z is continuous so the 
problem becomes

$$
\int_z p(\bar x, z; \theta)dz
$$

So we have a dataset $\mathcal D$ where for each datapoint variables are observed
like pixels. We also for each datapoint have unobserved latent variables such
as the semantic meaning of brown hair, or cluster ids which are never observed.

To learn the distribution we would still like to use the maximum likelihood.
(because as before this was derived from the KL divergence for distance
between our data distribution and our real distribution).

$$
\log \prod_{x \in \mathcal D}p(x; \theta) 
= \sum_{x \in \mathcal D} \log p(x; \theta) 
= \sum_{x \in \mathcal D} \log \sum_z  p(x, z; \theta)
$$
or even 
$$
= \sum_{x \in \mathcal D} \log \int_z p(x, z; \theta) dz
$$

which quickly becomes intractable

One attempt to solve this is to use **Importance Sampling** where we attempt to
weight the zs by a different distribution q(z), from which we end up sampling

$$
p_\theta(x) = \sum_z p_\theta(x, z) 
= \sum_{z \in \mathcal Z} \frac{q(z)}{q(z)} p_\theta(x, z)
= \mathbf E_{z \sim q(z)}\left[\frac{p_\theta(x,z)}{q(z)}\right]
$$

for which we then can use monte carlo 

$$
p_\theta(x) \approx \frac{1}{k} \sum^k_{j=1} \frac{p_\theta(x,z)}{q(z)}
$$

Now you need to find a good q. The task becomes to find a q that gives likely 
zs for x

This now is an unbiased estimator of $p_\theta(x)$

$$
\mathbf E_{z^{(j)}\sim q(z)}
\left[ 
    \frac{1}{k} \sum^k_{j=1} \frac{p_\theta(x,z)}{q(z)} 
\right]
= p_\theta(x)
$$

however because we want the log-likelihood $\log p_\theta(x)$
so 

$$
\log p_\theta(x) \approx \log 
\left( 
    \frac{1}{k} \sum^k_{j=1} \frac{p_\theta(x,z)}{q(z)} 
\right)
\stackrel{k=1}{\approx} \log
\left( 
    \frac{p_\theta(x,z^{(1)})}{q(z^{(1)})} 
\right)
$$

this is not an unbiased estimator

$$
\mathbf E_{z^{(1)}\sim q(z)}
\left[  
    \log \left( \frac{p_\theta(x,z^{(1)})}{q(z^{(1)})}  \right) 
\right]
\neq
\log
\left(  
    \mathbf E_{z^{(1)}\sim q(z)} 
    \left[ 
        \frac{p_\theta(x,z^{(1)})}{q(z^{(1)})}
    \right]
\right)
$$

now because $\log$ is a concave function we know that

$$
\mathbf E_{z^{(1)}\sim q(z)}
\left[  
    \log \left( \frac{p_\theta(x,z^{(1)})}{q(z^{(1)})}  \right) 
\right]
\leq
\log
\left(  
    \mathbf E_{z^{(1)}\sim q(z)} 
    \left[ 
        \frac{p_\theta(x,z^{(1)})}{q(z^{(1)})}
    \right]
\right)
$$

making the first term a lower bound which we can optimize. We also call it the
**Evidence lower bound (ELBO)** Meaning its a lower bound to the probability of
the evidence x

### Variational Inference

The nice part is that this holds for any q

$$
\log p(x; \theta) \ge \sum_z q(z) \log \
\left(
    \frac{p_\theta (x,z)}{q(z)}
\right)
$$
$$
= \sum_z q(z) \log p_\theta (x,z) 
\underbrace{- \sum_z q(z) \log q(z)}_{\text{Entropy } H(q) \text { of } q}
$$
$$
= \sum_z q(z) \log p_\theta (x,z) + H(q)
$$
Which is what we compute in the **Expectation Maximization (EM)** algorithm

## Lecture 6 - VAEs 2

Lets examine once more the ELBO which holds for any q

$$
\underbrace{\sum_z q(z) \log p_\theta (x,z)}_{\text{Log Likelihood as if 
everything was fully observed}}
+ H(q)
$$

if $q = p(z|x;\theta)$ then 

$$
\log p(x;\theta) = \sum_z q(z) \log p_\theta (x,z) + H(q)
$$

which is exactly what we would like.


Another way to derive this
$$
D_{\text{KL}}(q(z)||p(z|x;\theta)) 
= - \sum_z q(z) \log p_\theta (x,z) + \log p_\theta(x) - H(q) \ge 0
$$
$$
    \log p_\theta(x) \ge \sum_z q(z) \log p_\theta (x,z) + H(q)
$$
and if $q = p(z|x;\theta)$ then 
$ D_{\text{KL}}(p(z|x;\theta)||p(z|x;\theta)) = 0$

So now it becomes the task to find a good q(z) such that it approximates
$p(z|x;\theta)$, which would essentially be the reverse of the decoder
$p(x|z;\theta)$. That makes it fairly obvious that computing this is not 
tractable. \
So lets instead assume, that $q_\phi(z)$ is a distribution over the latents 
parameterized by $\phi$ 
for example $q_\phi(z) = \mathcal N (\mu_\phi, \Sigma_\phi)$

Optimizing q such that it is as close as possible to $p(z|x;\theta)$ is called
**variational inference**

$$
\log p_\theta(x) \ge \sum_z q_\phi(z) \log p_\theta (x,z) + H(q_\phi(z))
= \underbrace{\mathbf L_{\phi, \theta}(x)}_{\text{ELBO}}
$$
it follows that 
$$
\log p_\theta(x) = \mathbf L_{\phi, \theta}(x) 
+ D_\text{KL}(q_\phi(x)||\log p_\theta(z|x))
$$

The choice of the latent variables z depends a lot on the given input. So
so ideally youd have a different distribution for each x in the dataset. This
is obviously true because the latents heavily depend on the observation

Therefore with l being the likelihood
$$
\max_\theta \mathcal l(\theta; \mathcal D) \ge 
\max_{\theta, \phi^i, ..., \phi^m} \sum _{x^i\in \mathcal D}
\mathbf L_{\theta,\phi^i}(x^i)
$$

so now lets train this using stochastic gradient descent

$$
\mathbf L_{\theta,\phi^i}(x^i) 
= \sum_z q_{\phi^i}(z) \log p_\theta(x^i,z)+H(q_{\phi^i}(z))
$$
$$
= \mathbf E_{q_{\phi^i}(z)}
\left[
    \log p_\theta(x^i,z) - \log q_{\phi^i}(z)
\right]
$$

then the steps according to **stochastic variational inference** would be
- Initialize $\theta, \phi^i, ..., \phi^m$
- Sample $x^i \sim \mathcal D$
- optimize $\mathbf L_{\theta,\phi^i}(x^i)$ as a function of $\phi^i$
- - Repeat $\phi^i = \phi^i + \eta\nabla_{\phi^i}L_{\theta,\phi^i}(x^i)$ 
- - until convergence to $\phi^{i,*}\approx \argmax_\phi L_{\theta,\phi}(x^i)$
- You now computed the optimal $\phi^i$ for this data point which is why you 
denote it *
- Compute $\nabla_{\theta}L_{\theta,\phi^{i,*}}(x^i)$

Key assumption is that q is tractable, i.e. easy to sample from and easy to 
evaluate

Now for calculating the gradients. $\theta$ is easy with the monte carlo log 
likelihood of the decoder. we omit the p(z) because we approximate the
expectation with monte carlo

$$
\nabla_\theta \mathbf E_{q_{\phi}(z)}
\left[
    \log p_\theta(x,z) - \log q_{\phi}(z)
\right]
= \nabla_\theta \mathbf E_{q_{\phi}(z)}
\left[
    \log p_\theta(x,z)
\right]
\approx \frac{1}{k}\sum_k \nabla_\theta \log p_\theta(x,z^k)
$$

Calculating the gradients of $\phi$ however is not so easy. There is a good
albeit less general approachg that only works for some continuous distributions
for q

### Reparameterization

We want to compute a gradient with respect to $\phi$ of

$$
\mathbf E_{q_{\phi}(z)}\left[ r(z) \right]
= \int q_\phi(z)r(z) dz
$$

where r is an arbitrary function and z is  **now continuous**

Suppose $q_\phi(z) = \mathcal N (\mu, \sigma^2 I)$ is Gaussian with parameters
$\phi = (\mu, \sigma)$ then these are equivalent ways of sampling:
- $z \sim q_\phi(z)$
- $\epsilon \sim \mathcal N (0, I), z = \mu + \sigma \epsilon 
= g_\phi(\epsilon) $

The advantage of the later approach is that we are not taking the gradient of a
non-deterministic function anymore but rather of g which is deterministic, and
takes a random variable which does not depend on the optimization parameters
as a parameter.

Using this we can evaluate the expectation as one of two was
$$
\mathbf E_{q_{\phi}(z)}\left[ r(z) \right]
= \int q_\phi(z)r(z) dz
= \mathbf E_{\epsilon \sim \mathcal N (0, I)}\left[ r(g_\phi(\epsilon)) \right]
= \int \mathcal N (\epsilon)r(\mu + \sigma \epsilon)d\epsilon
$$
Now that phi is no longer part of the expectation parameter we can "push the 
gradient inside"

$$
\nabla_\phi \mathbf E_{q_{\phi}(z)}\left[ r(z) \right]
= \nabla_\phi \mathbf E_{\epsilon \sim \mathcal N (0, I)}
\left[ r(g_\phi(\epsilon)) \right]
= \mathbf E_{\epsilon \sim \mathcal N (0, I)}
\left[ \nabla_\phi r(g_\phi(\epsilon)) \right]
$$

With Monte Carlo

$$
\mathbf E_{\epsilon \sim \mathcal N (0, I)}
\left[ \nabla_\phi r(g_\phi(\epsilon)) \right] \approx
\frac {1}{k} \sum_k \nabla_\phi r(g_\phi(\epsilon^k))
$$

Now that we know how to compute the gradient of both $\phi$ and $\theta$ we
only have to deal with the issue that we need a $\theta^i$ for every data point
which is not scalable \
Instead we now use **Amortization** where we learn a single parametric function
$f_\lambda$ that maps each x to a set of good variational parameters like doing
regression on $x^i \rightarrow \phi^{i,*}$   
So we approximate the posteriours $q(z|x^i)$ using the distribution 
$q_\lambda(z|x)$

So now essentially we replace $q_{\phi^i}(z)$ with $q_{f_\lambda (x^i)}(z)$
which in literature may also be denoted as $q_\phi(z|x)$

now you just have to 
- Compute $\nabla_{\theta}L_{\theta,\phi}(x^i)$ and 
$\nabla_{\phi}L_{\theta,\phi}(x^i)$ using both monte carlo and
reparemeterization for q if it is continuous such as a gaussian

## Lecture 7 - Normalizing Flows






