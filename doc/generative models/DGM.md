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
\text{prior} = P(z) = \vec z \sim \mathcal N (0, I) 
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

Let's once more examine what we are actually doing and why this is indeed an 
autoencoder

$$
\mathbf L_{\phi, \theta}(x) = \mathbf E_{q_{\phi}(z)}
\left[
    \log p_\theta(x,z) - \log q_{\phi}(z)
\right]
$$
$$
 = \mathbf E_{q_{\phi}(z)}
\left[
    \log p_\theta(x,z) 
    \underbrace{- \log p(z) + \log p(z)}
    _{\log p(z) \text{ is just a simple distribution such as }\mathcal N} 
    - \log q_{\phi}(z)
\right]
$$
$$
\mathbf L_{\phi, \theta}(x) = \mathbf E_{q_{\phi}(z)}
\left[
    \log p_\theta(x|z) - D_\text{KL} (\log q_{\phi}(z|x) || p(z))
\right]
$$

The first term is the feature of an autoencoder, it aims to enforce, that our
learnt distribution $p_\theta(x)$ given the latent features is able to create a
good reconstruction of x by making it very likely.

The KL Divergence in the second term enforces, that the learnt latents resemble
the latents we want to use for generation. For example a simple gaussian. This
ensures, that our latents are actually in form of a distribution from which we
can sample


## Lecture 7 - Normalizing Flows

So far weve seen two model families:
- Autoregressive models: $p_\theta(x)=\prod_{i=1}^n p_\theta(x_i|x_{<i})$
- Variational Autoencoders: $p_\theta(x)=\int p_\theta(x,z)dz$

Autoregressive Models are easy to sample from and to train, however they dont 
have a good way to learn features, whereas VAEs learn those features but have
intractible marginal likelihoods. Is there a way to combine the advantages of 
both?\
The Idea behind flow models is that they transform simple distributions like
gaussians to complex functions, through an **invertible transformation**

A flow model is similar to a VAE
- We again start with a simple prior $P(z) = \vec z \sim \mathcal N (0, I)$
- Now instead of having a very expensive integral to compute such as 
$p_\theta(x)=\int p_\theta(x,z)dz$ could we easily invert $p_\theta(x|z)$ and
compute $p_\theta(z|x)$ by design - Make $x = f_\theta(z)$ a deterministic and
invertible function of z so that for anz x there is a unique z. So each z
produces one and only one x and vice versa, so it becomes easy to compute because
we dont need an integral over all zs that could have produces x as its only 1

### Continuous random variable refresher

X is a continuous random variable

The cumulative density function CDF of X is $F_X(a) = P(X\le a)$

The probability density function PDF of X is 
$p_X(a) = F'_X(a) = \frac{dF_X(a)}{da}$

**Change of Variables in 1D**: If $X = f(Z)$ and $f(.)$ is monotone with inverse
$Z = f^{-1}(X) = h(X)$ then
$$
p_X(x)=p_Z(h(x))|h'(x)|
$$
So this just means if you transform a random variable to another the underlying
density function for the new random variable is not just the transformed 
original density, but the scaling applied for the entire domain.\
To clarify this once more, the first term simply maps the point x from the 
distribution $p_Z$ to the distribution $p_x$ and the derivative of that mapping
function gives us the scale of the transformation at that particular point. 
This is easy to see because the derivative is simply the local rate of change 
which is just the local scaling of the one distribution when mapped to the 
other

Using this knowledge about the derivative of an inverse

$$
(f^{-1})'(x) = \frac{1}{f'(f^{-1}(x))}
$$

we can change the change of variables formula to 
$$
p_X(x) = \frac{p_Z(z)}{f'(z)}
$$

This is the case in 1d but luckily the multidimensional case is not too different:

Let Z be a uniform random vector and let X = AZ for a square (invertible, given
by that it is square) matrix and the inverse $W=A^{-1}$\
Because a matrix is a linear mapping A would map the uniform distribution in n
dimensions (which would be some form of hypercube) to a linearly stretched form
of that hypercube, which we also call a parallelotope.

The Volume of the parallelotope happens to be equal to the absolute value of the 
determinant of A

so in the simple case of a uniform distribution with a linear mapping it 
happens that 
$$
p_X(x) = \frac{p_Z(Wx)}{|\det A|}
$$
In fact the distribution is arbitrary

Now we need to consider, that we dont just want a linear transformation for x
we want a non linear neural network. Luckily this does not change the way the 
mapping translates to the different distribution i.e. the first term and it
only affects the local scalings of the distribution mapping. So now rather than
being able to simply take the determinant of A we have to take the determinant
of the jacobian matrix of the transformation f. This is because in the case
of a linear transformation the scaling of course is linear so it doesnt depend
on the mapping region, however with an arbitrary scaling we have to examine the
local scalings, i.e. the local rates of change, i.e. the derivatives, i.e. the
jacobian matrix.

this leads us to the **General Case Change of Variables**: 
$$
p_X(x) = p_Z(f^{-1}(x))
\left| \det \left(
    \underbrace{\frac{\partial f^{-1}(x)}{\partial x}}_\text{Jacobian Matrix}    
\right) \right|
$$
this also is the same as
$$
p_X(x) = p_Z(f^{-1}(x))
\left| \det \left(
    \frac{\partial f(z)}{\partial z}   
\right) \right| ^{-1}
$$
Note that in order to be able to invert f x and z have to be of the same
dimensionality

## Lecture 8 - Normalizing Flows

A **flow** is a composition of invertible transformations
$$
z_m = f^m_\theta \circ ... \circ f^1_\theta (z_0) 
\stackrel{\Delta}{=} f_\theta (z_0)
$$

By change of variable 

$$
p_X(x; \theta) = p_Z(f^{-1}_\theta(x))\prod_{m=1}^M
\left| \det \left(
    \frac{\partial (f_\theta^m)^{-1}(z^m) }  {\partial z^m}  
\right) \right|
$$

$z^M$ being the mapping for x and each previous ones being steps from the 
original z

Sampling, just like VAEs is very simple $z \sim p_Z(z), x = f_\theta(z)$

Now in practice what do we want from a flow model?
- a simple prior $p_Z(z)$ that is easy to sample from and has a tractable
likelihood evalutation
- invertible functions with tractable evaluation
- We need to compute the jacobian matrices
- - computing the determinant for an nxn matrix is $O(n^3)$
- - Key Idea: Choose transformations, so that the resulting jacobian matrix
has a special structure. For example the determinant of a triangular matrix is
the product of the diagonal entries. Triangular means that every entry above
the main diagonal is 0

you can achieve a triangular matrix by enforcing that each function can only
depend on input components before it (or after)! Similar to how an 
autoregressive model can say generate an image only on the basis of pixels it
so far generated

$$
x=(x_1, ..., x_n) = f(z) = (f_1(z), ..., f_n(z))
$$
$$
J = \frac{\partial f}{\partial z} = 
\left(
\begin{matrix}

\frac{\partial f_1}{\partial z_1} & \cdots & \frac{\partial f_1}{\partial z_n} \\
\vdots                            & \ddots & \vdots \\
\frac{\partial f_n}{\partial z_1} & \cdots & \frac{\partial f_n}{\partial z_n} \\

\end{matrix}
\right)
$$

now if $x_i = f_i(z)$ only depends on $z_{\le i}$ then

$$
\left(
\begin{matrix}

\frac{\partial f_1}{\partial z_1} & \cdots & 0\\
\vdots                            & \ddots & \vdots \\
\frac{\partial f_n}{\partial z_1} & \cdots & \frac{\partial f_n}{\partial z_n} \\

\end{matrix}
\right)
$$

---

### Continuous Autoregressive models as flow models

Consider a gaussian autoregressive model

$$
p(x) = \prod_1^n p(x_i|x_{<i})
$$

such that $p(x_i|x_{<i}) = 
\mathcal N(\mu_i(x_1,...,x_{i-1}), \exp(\alpha(\mu_i(x_1,...,x_{i-1})))$ where 
$\mu$ and $\alpha$ are neural networks

- Sample $z \sim \mathcal N(0,1)$
- let $x_1 = \exp (\alpha_i) z_1 + \mu_1$ Compute $\mu_2(x_1), \alpha_2(x_1)$
- - etc etc

Which you can tell is very similar to a flow model in fact this model is called
Masked autoregressive flow **MAF**

Useful thing about this is that you can learn everything in parallel because 
you have all the xs

You can even invert this process, which is then called an inverted autoregressive
flow, which just means that we take all the latents as given or the xs and
training becomes slow, whereas sampling now is fast **IAF**

You can try to get the best of both worlds with a **teacher student model**
where the teacher is an MAF once a teacher is trained you initialize a student
which is parameterized by IAF which you then try to train by minimizing the KL
Divergence between the two


## Lecture 9 - GANs

The idea of all the above models is to assign probabilities to each data point.
This is to be able to use likelihood maximization in order to train them. Or
equivalently minimizing the KL Divergence between the model and the data 
distribution.

It can actually be shown that this is the fastest method to train a statistical 
model with the optimal statistical efficiency. I.e. $\hat \theta$ converges
to $\theta^*$ when the amount of data approaches infinity the fastest. 

However high likelihood is not necessarily the best indicator of the quality
of the generated samples

One example where this is illustrated is the below model

$$
\log p_\theta(x) = \log[0.01p_\text{data}(x)+0.99p_\text{noise}(x)]
$$

So this just adds 1 percent of the image + 99 percent pure noise

$$
\log p_\theta(x) = \log[0.01p_\text{data}(x)+0.99p_\text{noise}(x)]
\ge \log 0.01p_\text{data}(x) = \log p_\text{data}(x) - \log 100
$$

lower bound
$$
\mathbf E_{p_\text{data}} [\log p_\theta(x)] 
\ge \mathbf E_{p_\text{data}} [\log p_\text{data}(x)] - \log 100
$$

upper bound
$$
\mathbf E_{p_\text{data}} [\log p_\theta(x)] 
\ge \mathbf E_{p_\text{data}} [\log p_\text{data}(x)]
$$

As we increase the dimension of the likelihood of the data for all images
increases proportionally to the dimensions, because we just sum them up. 
Whereas the constant remains a constant. So this will have a near perfect
likelihood with high dimensional data.

Similarly the opposite can be true, where the log likelihood is awful but the 
quality of the samples is fantastic. For example by complete overfitting

And thats what we try to do with GANs, we dont use likelihood to learn

So we will be looking at alternatives for $d(p_\text{data}, p_\theta)$

One way to do this is to compare the samples of two distributions. Figure out a
way to tell if a sample belongs to distribution A or B: \
Given the distributions and samples $S_1 = {x \sim P}$ and $S_2 = {x \sim Q}$
a **two-sample test** considers the following hypothesis

- Null hypothesis $H_0: P = Q$
- Alternative hypothesis $H_1: P \ne Q$

Test statistic T compares $S_1$ and $S_2$. For example the difference in mean 
and variance of the two samples

$$
T(S_1, S_2) = 
\left| 
    \frac{1}{|S_1|}\sum_{x \in S_1} x
    - \frac{1}{|S_2|}\sum_{x \in S_2} x
\right|
$$

Then you can define some decision boundry that rejects $H_0$ at a certain 
threshold, otherwise we say $H_0$ is consistent with the observation

What is important to notice is that this Test statistic does not include 
likelihoods at all. There are no densities in the measure itself only samples \
however the problem is that finding a good test statistic in high dimensions is 
not easy. -> Learn how to do it with a neural network. This we also call a
discriminator, which is really just a classifier network $D_\phi$.\
Now our test statistic becomes the -loss of the classifier. This is because
when the loss is low, the classifier is doing a really good job, which means
the samples are easily not the same. Otherwise if the loss is high, we expect
the samples to be very close to each other

The goal of the classifier is to maximize the test statistic, or equivilantly
minimize the classification loss

Training for the discriminator

$$
\max_{D_\phi} V(p_\theta, D_\phi) 
= \mathbf E_{x \sim p_\text{data}} [\log D_\phi (x)]
+ \mathbf E_{x \sim p_\theta} [\log (1 - D_\phi (x))]
$$

which is basically just standard cross entropy

Optimal Discriminator:
$$
D^*_\theta = \frac{p_\text{data}(x)}{p_\text{data}(x) + p_\theta (x)}
$$

This has to be true as if our data distribution is equal to the generated 
distribution, the optimal classification is exactly $\frac{1}{2}$ Also note
that this is not a neural network, it is using the base distributions. So this
is really just an optimal construct that already knows if something was data
or generated.

Now when you introduce the generator it becomes a minimax optimization game,
because the generator tries to minimize the second training term

Generator
- Sample z  from a simple prior like all the other latent models
- Set $x = G_\theta(z)$

We can use any neural network to model the data distribution now, because it is
not bound to a likelihood. We dont need likelihoods because we simply dont care
that a given data image has high likelihood under our distribution. All we care
out is that its hard to distinguish between the generated sample and the data 
sample.

$$
\min_{G_\theta} \max_{D_\phi} V(G_\theta, D_\phi) 
= \mathbf E_{x \sim p_\text{data}} [\log D_\phi (x)]
+ \mathbf E_{x \sim p_{G_\theta}} [\log (1 - D_\phi (x))]
$$

Now if we plug in our optimal discriminator 

$$
V(G_\theta, D_\phi^*) 
= \mathbf E_{x \sim p_\text{data}} 
\left[
    \log \frac{p_\text{data}(x)}{p_\text{data}(x) + p_{G_\theta}(x)}
\right]
+ \mathbf E_{x \sim p_{G_\theta}} 
\left[ 
    \log \frac{p_{G_\theta}(x)}{p_\text{data}(x) + p_{G_\theta}(x)}
\right]\\
= \mathbf E_{x \sim p_\text{data}} 
\left[
    \log  
    \frac{p_\text{data}(x)}
    {\frac{p_\text{data}(x) + p_{G_\theta}(x)}{2}}
\right]
+ \mathbf E_{x \sim p_{G_\theta}} 
\left[ 
    \log  
    \frac{p_{G_\theta}(x)}
    {\frac{p_\text{data}(x) + p_{G_\theta}(x)}{2}}
\right] - \log 4\\
= \underbrace{D_\text{KL}
\left[
    p_\text{data} || \frac{p_\text{data} + p_{G_\theta}}{2}
\right]
+ D_\text{KL}
\left[
    p_{G_\theta} || \frac{p_\text{data} + p_{G_\theta}}{2}
\right]}_\text{2 times Jensen-Shannon Divergence (JSD)} - \log 4 \\

= 2D_\text{JSD}[p_\text{data}, p_{G_\theta}] - \log 4
$$

### Jensen-Shannon Divergence

aka symmetric KL Divergence

$$
D_\text{JSD}[p,q]=\frac{1}{2}
\left(
D_\text{KL} \left[p || \frac{p+q}{2}\right]
+ D_\text{KL} \left[q || \frac{p+q}{2}\right]
\right)
$$

So its the distance between a mix of p and q with respect to both p and q. The
properties of the JSD is that it is always positive, 0 iff p = q and 
symmetrical (also $\sqrt{D_\text{JSD}[p,q]}$ satisfies triangle inequality)

Pros and cons of GAN
- + Loss does not require likelihood, so theres virtually no restrictions to 
the generator
- + very flexible network architecture as long as we have a valid sampling 
procedure
- + fast sampling

- - however it is very difficult to train in practice due to the minimax 
problem

Some of these problems included
- Instability due to minimax
- You dont really know when to stop, because theres no clear objective to
converge to like in kl divergence
- Mode Collapse, which basically means the generator only focuses on a small
sub group of the data to generate and not the entire dataset

That in mind discriminator training is still quite powerful when in combination
with other approaches

## Lecture 10 - GANs 

### f divergences

Given two densities p and q, the f divergence is given by 

$$
D_f(p,q) = E_{x\sim q}\left[ f(\frac{p(x)}{q(x)}) \right]
$$

where f is any convex, lower semicontinous function with f(1) = 0, so something 
like -log, so its kinda like a broader version of the kl divergence

- Convex: Line joining any two points lies above the function
- lower-semicontinuous: function value at any point x is close to or greater 
than it function value

The f(1) = 0 invariant makes sense, because that means p and q are the same
so the distance between them has to be 0

Jensens inequality: $\mathbf E_{x \sim q} \left[ f(\frac{p(x)}{q(x)}) \right] 
\ge f(\mathbf E_{x \sim q} \left[ \frac{p(x)}{q(x)} \right]) 
= f(\int q(x) \frac{p(x)}{q(x)})= f(\int p(x)) = f(1) = 0$

which proves that its also always positive if the distributions are well defined

(you might notice that if the function is concave the reverse is true)

As weve seen before we dont want to compute the divergence between the data
distribution and the model distribution. Ideally not even just the model 
dsitribution as that requires the model to model likelihoods, which GANs for
example dont do in general. As for the data distribution its obvious that
we dont even have that, which is why we had to use the lower bound in the
original example 

### towards variational divergence minimization 

for any function f its convex conjugate is
$$
f^*(t) = \sup_{u \in \text{domain}_f}(ut-f(u))
$$

sup = supremum the upper bound (so with non divergent functions the global
maximum) as opposed to the opposite which is the infimum

This guarantees that $f^*$ is convex and lower semi continuous

now f** is the **Fenchel conjugate** which is the conjugate of f

this has the property, that $f^{**}\le f^*$ \
proof:
$$
f^*(t) \ge ut-f(u) \equiv f(u) \ge ut-f^*(t)\\
f(u) \ge \sup_t(ut-f^*(t)) = f^{**}(u)
$$

when f is convex and lower semicontinuous the fenchel conjugate of f = f

### f-GAN

we obtain a lower bound to an f divergence via Fenchel conjugate

$$
\mathbf E_{x \sim q}\left[ f(\frac{p(x)}{q(x)}) \right] 
= \mathbf E_{x \sim q}\left[ f^{**}(\frac{p(x)}{q(x)}) \right]\\
= E_{x \sim q}\left[\sup_{t \in \text{domain}_f^*}(t\frac{p(x)}{q(x)}-f^*(t))
\right]\\
= E_{x \sim q}\left[T^*(x)\frac{p(x)}{q(x)}-f^*(T^*(x))
\right]\\
= \int_x q(x)\left[ T^*(x)\frac{p(x)}{q(x)}-f^*(T^*(x))  \right] dx \\
= \int_x q(x)\left[ T^*(x)p(x) - f^*(T^*(x))q(x)  \right] dx \\
= \sup_T \int_x q(x)\left[ T(x)p(x) - f^*(T(x))q(x)  \right] dx \\
\ge \sup_{T\in \mathcal T} \int_x q(x)\left[ T(x)p(x) - f^*(T(x))q(x)  \right] dx
| \text{where } \mathcal T : \mathcal X \mapsto \mathbb R 
\text{ is an arbitrary class of functions} \\
= \sup_{T\in \mathcal T} (\mathbf E_{x\sim p}\left[T(x)\right] 
- \mathbf E_{x\sim q}\left[f^*(T(x)\right)])
$$


We can replace the supremum operator with T(x) because we know that theres 
going to be a different density depending on the x, which also means that the 
optimal value of t depends on x, which is why we can replace it with a funciton
with respect to X\
The lower bound works, because you are saying T now has to be part of a subset
of all functions, which of course has to be less or equal to an optimal function 
which is part of the set of all functions. And the more flexible that family for
T is the better the lower bound will be

Applying this to the concept of GANs you can tell that T is the discriminator.

So now you have an arbitrary variational lower bound for any f divergence
$$ D_f(p,q) \ge \sup_{T\in \mathcal T} (\mathbf E_{x\sim p}\left[T(x)\right] 
- \mathbf E_{x\sim q}\left[f^*(T(x)\right)])
$$

From this you get the f-GAN training objective
$$
\min_\theta \max_\phi F(\theta, \phi) = 
\mathbf E_{x\sim p_\text{data}}\left[T_\phi(x)\right] 
- \mathbf E_{x\sim p_{G_\theta}}\left[f^*(T_\phi(x)\right)]
$$

### Wasserstein-GAN

The support of q has to cover the support of p. Otherwise discontinuity arises
in f divergences. i.e. when the generator generates samples that are very
different from the dataset. This means that the divergence has a very high 
value until it somewhat resembles the data and snaps to a better value. 

(support means the set of values a Random Variable can take with non zero
probability)

This is bad because it doesnt give you a good signal to backpropagate

for example take these two distributions
$$
p(x) = \begin{cases}
1,  & x=0 \\
0, & x\ne0
\end{cases}
$$
$$
q_\theta(x) = \begin{cases}
1,  & x=\theta \\
0, & x\ne\theta 
\end{cases}
$$
then
$$
D_\text{KL}(P||Q) = \begin{cases}
0,  & \theta = 0 \\
\infty, & \theta \ne 0 
\end{cases}
$$

$$
D_\text{JSD}(P||Q) = \begin{cases}
0,  & \theta = 0 \\
\log 2, & \theta \ne 0 
\end{cases}
$$

So this calls for a better Distance function that is defined when p and q have
disjoint supports. And one way to  do this is with the Wasserstein distance

If you think about a distribution as a pile of earth and you compare two piles 
of earth. you could ask yourself how much effort would it take to move the one
set of piles so that it resembles the other. Intuitively the further you have
to move the dirt the more effort that is.

Wasserstein distance:
$$
D_w(p,q) = \inf_{\gamma \in \Pi(p,q)}\mathbf E_{(x,y)\sim\gamma}
[\Vert{x-y}\Vert_1]
$$

Where $\Pi(p,q)$ contains ALL (possible) joint distributions of (x,y) where 
the marginal of x is $p(x)=\int\gamma(x,y)dy$ and the marginal of y is $q(y)$\
(Gamma is a joint distribution)

So what this means is that from the set of joint distributions you only take 
those where the marginals of x match p(x) and the marginals of y match (y)
and you then take the joint distribution that has the smallest l1 distance
over the expected values respectively of the joint distribution gamma

in this case p would be defined over x and q over y
$$
p(x) = \begin{cases}
1,  & x=0 \\
0, & x\ne0
\end{cases}
$$
$$
q_\theta(x) = \begin{cases}
1,  & x=\theta \\
0, & x\ne\theta 
\end{cases}
$$
then
$$
D_\text{W}(P||Q) = |\theta|
$$

This runs into issues because we are working with probability distributions over
the model and the data, which is not something we can really compute
which we are trying to optimize. So instead we use a variational called the 
**Kantorovich-Rubinstein duality**

$$
D_w(p,q) = \sup_{\Vert f\Vert_L \le 1}\mathbf E_{x \sim p}[f(x)]
 -\mathbf E_{x \sim q}[f(x)]
$$

$\Vert f\Vert_L \le 1$ means the **Lipschitz constant** of f(x) is 1

$$
\forall x,y: |f(x)-f(y)|\le \Vert x-y \Vert_1
$$

So the distance between x and y has to be larger than the distance of their
function values for all possible x and ys

$$
\min_{G_\theta} \max_{D_\phi} V(G_\theta, D_\phi) 
= \mathbf E_{x \sim p_\text{data}} [\log D_\phi (x)]
- \mathbf E_{x \sim p_{G_\theta}} [ D_\phi (G_\theta(x)))]
$$

Then you can enforce the Lipschitzness through weight clipping in the 
discriminator or gradient penalties on it

This is more stable in practice than optimizing JSD

### Is there a way to get Latent representations in a gan

currently the problem is that there is no direct connection between x and z

One solution is to use the activations of the prefinal layer of a discriminator
as a feature representation

The intuition behind this is that in order to discriminate between fake and
real data you have to be able to learn interesting features over the dataset\
However if you want to learn to infer the latent variables of the generator 
you need a different learning algorithm. this is because a regular gan optimizes
the two sample test that compares x and the generated samples. It does not 
actually include the latents. In fact the data is not connected to the latents
as in its not a hidden conditional

So the solution in regards to this problem would be to compare samples from 
x and z from the joint distribution of observed and latent variables as per the 
model and the data distribution

Looking at BiGAN as a way to solve this is that the discriminator does not
only distinguish generated and real data samples, but also generated and real
latent representations. To this end you introduce a new component, the encoder,
that maps a data sample to a latent representation

At inference time you can then sample from the generator, and if you need 
latents of real data, use the encoder


## Lecture 11 - Energy Based Models

Tries to fix the stability issues of gans, while still preserving the high
sample quality and flexible architecture freedom

A key concept to generative models are probability distributions $p(x)$
These distributions have certain invariants
- $p(x) \ge 0$
- $\int p(x) dx = 1$

Especially the last invariant is very important, as it ensures that optimizing
likelihoods actually improves the model quality. Because what increasing the 
likelihood should mean that while this is now more likely, everything else
should be less likely. Whereas if you just increase the likelihood indefinitely
it loses its original purpose.\
So this is important to ensure, but unfortunately its difficult.

Problem:\
Let g be a non negative function for any x. g need not be normalized. How do
you generally normalize it?

$$
p_\theta(x) = \frac{1}{Z(\theta)}g_\theta(x)
= \frac{1}{\int g_\theta(x) dx}g_\theta(x)
= \frac{1}{\text{volume}(g_\theta)}g_\theta(x)
$$

Then by definition $\int p_\theta(x) dx = 1$






