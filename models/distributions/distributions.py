import math
import numpy as np
import theano.tensor as T

# ----------------------------------------------------------------------------
# this is all taken from the parmesan lib

c = - 0.5 * math.log(2*math.pi)

def log_gumbel_softmax(x, mu, tau=1.0, eps=1e-6):
    """
    Compute logpdf of a Gumbel Softmax distribution with parameters p, at values x.
        .. See Appendix B.[1:2] https://arxiv.org/pdf/1611.01144v2.pdf
    """
    k = mu.shape[-1]
    logpdf = T.gammaln(k) + (k - 1) * T.log(tau + eps) \
        - k * T.log(T.sum(T.exp(mu) / T.power(x, tau), axis=2) + eps) \
        + T.sum(mu - (tau + 1) * T.log(x + eps), axis=2)
    return logpdf

def log_logistic_sigmoid(x, mu, tau=1.0, eps=1e-6):
    """
    Compute logpdf of a Gumbel Softmax distribution with parameters p, at values x.
        .. See Appendix B.[1:2] https://arxiv.org/pdf/1611.01144v2.pdf
    """
    mu = T.clip(mu, -10., 10.)
    logpdf = mu + T.log(tau + eps) \
        - (tau+1.) * ( T.log(x + eps) + T.log( 1.-x + eps) ) \
        - 2. * T.log( T.exp(mu) * T.power(x,-tau) + T.power(1.-x,-tau) + eps )
    return logpdf

def log_logistic_sigmoid2(y, mu, tau=1.0, eps=1e-6):
    """
    Compute logpdf of a Gumbel Softmax distribution with parameters p, at values x.
        .. See Appendix B.[1:2] https://arxiv.org/pdf/1611.01144v2.pdf
    """
    mu = T.clip(mu, -10., 10.)
    logpdf = mu + T.log(tau + eps) - tau*y \
           - 2. * T.log( 1. + T.exp( -tau*y + mu ) + eps )
    return logpdf

def log_bernoulli(x, p, eps=1e-6):
    """
    Compute log pdf of a Bernoulli distribution with success probability p, at values x.
        .. math:: \log p(x; p) = \log \mathcal{B}(x; p)
    Parameters
    ----------
    x : Theano tensor
        Values at which to evaluate pdf.
    p : Theano tensor
        Success probability :math:`p(x=1)`, which is also the mean of the Bernoulli distribution.
    eps : float
        Small number used to avoid NaNs by clipping p in range [eps;1-eps].
    Returns
    -------
    Theano tensor
        Element-wise log probability, this has to be summed for multi-variate distributions.
    """
    p = T.clip(p, eps, 1.0 - eps)
    return -T.nnet.binary_crossentropy(p, x)


def log_categorical(x, p, eps=1e-6):
    """
    Compute log pdf of a Categorical distribution with success probability p, at values x.
        .. math:: \log p(x; p) = \log \mathcal{B}(x; p)
    Parameters
    ----------
    x : Theano tensor
        Values at which to evaluate pdf.
    p : Theano tensor
        Success probability :math:`p(x=1)`, which is also the mean of the Bernoulli distribution.
    eps : float
        Small number used to avoid NaNs by clipping p in range [eps;1-eps].
    Returns
    -------
    Theano tensor
        Element-wise log probability, this has to be summed for multi-variate distributions.
    """
    p = T.clip(p, eps, 1.0 - eps)
    return -T.nnet.categorical_crossentropy(p, x)    


def log_normal(x, mean, std, eps=1e-6):
    """
    Compute log pdf of a Gaussian distribution with diagonal covariance, at values x.
    Variance is parameterized as standard deviation.
        .. math:: \log p(x) = \log \mathcal{N}(x; \mu, \sigma^2I)

    Parameters
    ----------
    x : Theano tensor
        Values at which to evaluate pdf.
    mean : Theano tensor
        Mean of the Gaussian distribution.
    std : Theano tensor
        Standard deviation of the diagonal covariance Gaussian.
    eps : float
        Small number added to standard deviation to avoid NaNs.
    Returns
    -------
    Theano tensor
        Element-wise log probability, this has to be summed for multi-variate distributions.
    See also
    --------
    log_normal1 : using variance parameterization
    log_normal2 : using log variance parameterization
    """
    std += eps
    return c - T.log(T.abs_(std)) - (x - mean)**2 / (2 * std**2)


def log_normal2(x, mean, log_var, eps=1e-6):
    """
    Compute log pdf of a Gaussian distribution with diagonal covariance, at values x.
    Variance is parameterized as log variance rather than standard deviation, which ensures :math:`\sigma > 0`.
        .. math:: \log p(x) = \log \mathcal{N}(x; \mu, \sigma^2I)

    Parameters
    ----------
    x : Theano tensor
        Values at which to evaluate pdf.
    mean : Theano tensor
        Mean of the Gaussian distribution.
    log_var : Theano tensor
        Log variance of the diagonal covariance Gaussian.
    eps : float
        Small number added to denominator to avoid NaNs.
    Returns
    -------
    Theano tensor
        Element-wise log probability, this has to be summed for multi-variate distributions.
    See also
    --------
    log_normal : using standard deviation parameterization
    log_normal1 : using variance parameterization
    """
    return c - log_var/2 - (x - mean)**2 / (2 * T.exp(log_var) + eps)
