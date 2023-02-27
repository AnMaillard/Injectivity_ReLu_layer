# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, fixme, line-too-long, missing-module-docstring, multiple-imports, pointless-string-statement

import jax.numpy as jnp
from jax import jit, vmap

"""
Auxiliary functions for the Full-RSB calculations.
Note that the JAX implementation of the FFT algorithm (which works on GPU) has error higher than a CPU-only version.
However this is to be expected for GPU vs CPU algorithms, and is not an issue at our level of precision.
"""

SQRT_2PI = jnp.sqrt(2.*jnp.float64(jnp.pi))

@jit
def gaussian_conv_sw_jax(f, var, H):
    #Does the convolution of f and a Gaussian of variance var. Here we assume var > 0
    N = int((jnp.size(f) - 1)//2) #In JAX, inner arrays can not be broadcasted wrt a parameter of the function, but shapes of parameters are OK
    N_f = jnp.float64(N) #Float value

    fhat, factor = jnp.zeros(2*N+1, dtype = 'complex128'), jnp.zeros(2*N+1, dtype = 'float64')
    result, convhat = jnp.zeros(2*N+1, dtype = 'float64'), jnp.zeros(2*N+1, dtype = 'float64')
    #We compute the FFT of f
    fhat = jnp.fft.fft(f)
    #Gaussian DFT using the Shannon-Whittaker sinc interpolation
    factor = jnp.exp(- (2*jnp.pi**2*var*(N_f/H)**2) *(jnp.arange(2*N+1)/ (2*N_f+1) - (jnp.arange(2*N+1) // (N+1)) )**2)
    convhat = jnp.multiply(fhat, factor)
    result = jnp.real(jnp.fft.ifft(convhat))
    #Functions are assumed to be real, and positive as well. We truncate all values <= threshold to threshold
    threshold = jnp.float64(1e-30)
    result_thresholded = jnp.maximum(threshold*jnp.ones_like(result),result)

    return result_thresholded

@jit
def gamma_jax(w, x):
    return jnp.exp(-x**2/(2*w))/(SQRT_2PI*jnp.sqrt(w))

@jit
def vmap_gamma_jax(w, v):
    return vmap(gamma_jax, (None, 0), 0)(w, v)

#This is f(x_k = inf, h) in the rescaled version
@jit
def f_inf(h):
    return jnp.where(h < 0, 0, jnp.where(h > 1, -1, -h**2) )

@jit
def vmap_finf(h):
    return vmap(f_inf)(h)
