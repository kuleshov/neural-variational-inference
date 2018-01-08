import numpy as np
import theano.tensor as T
import matplotlib.pyplot as plt

# MY CODE

temperature = 0.01
logits = np.linspace(-2, 2, 10).reshape([1, -1])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sample_gumbel(shape, eps=1e-20):
    """ Sample from Gumbel(0, 1) """
    U = np.random.uniform(size=shape, low=0, high=1)
    return -np.log(-np.log(U + eps) + eps)

def sample_gumbel_softmax(logits, temperature):
    """ Sample from Gumbel-Softmax distribution """
    y = logits + sample_gumbel(logits.shape)
    return softmax(y / temperature)

plt.title('gumbel-softmax samples')
for i in range(100):
    plt.plot(
        range(10),
        sample_gumbel_softmax(logits,temperature=temperature),
        marker='o',
        alpha=0.25
    )

plt.ylim(0,1)
plt.show()

gsm_samples = [sample_gumbel_softmax(logits,temperature=temperature) for _ in range(500)]
plt.title('average over samples')
plt.plot(
    range(10),
    np.mean(gsm_samples, axis=0)[0],
    marker='o',
    label='gumbel-softmax average'
)
plt.plot(
    softmax(sample_gumbel_softmax(logits,temperature=temperature)[0]),
    marker='+',
    label='regular softmax'
)
plt.legend(loc='best')
plt.show()

# ---------------------------------------------------------------------

# THEIR CODE
temperature = 0.01
logits = np.linspace(-2, 2, 10)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def GumbelSoftmax(logits, temperature):
    uniform = np.random.uniform(size=logits.shape,low=0,high=1)
    gumbel = -np.log(-np.log(uniform + 1e-20) + 1e-20)
    return softmax((logits + gumbel) / temperature)

plt.title('gumbel-softmax samples')
for i in range(100):
    plt.plot(
        range(10),
        GumbelSoftmax(logits, temperature),
        marker='o',
        alpha=0.25
    )

plt.ylim(0,1)
plt.show()

gsm_samples = [sample_gumbel_softmax(logits,temperature=temperature) for _ in range(500)]
plt.title('average over samples')
plt.plot(
    range(10),
    np.mean(gsm_samples, axis=0)[0],
    marker='o',
    label='gumbel-softmax average'
)
plt.plot(
    softmax(sample_gumbel_softmax(logits,temperature=temperature)[0]),
    marker='+',
    label='regular softmax'
)
plt.legend(loc='best')
plt.show()
