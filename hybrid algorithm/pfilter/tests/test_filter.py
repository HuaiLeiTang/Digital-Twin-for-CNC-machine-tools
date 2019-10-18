from pfilter import ParticleFilter, squared_error, gaussian_noise, cauchy_noise, make_heat_adjusted
import numpy as np


def test_init():
    # silly initialisation, but uses all parameters
    pf = ParticleFilter(
        prior_fn=lambda x: np.random.normal(0, 1, (1, 100)),
        observe_fn=lambda x: x,
        n_particles=100,
        dynamics_fn=lambda x: x,
        noise_fn=lambda x: x,
        weight_fn=lambda x: x,
        resample_proportion=0.2,
        column_names=["test"],
        internal_weight_fn=lambda x: x,
        n_eff_threshold=1.0,
    )

def test_gaussian_noise():
    np.random.seed(2012)
    for shape in [10,10], [100,1000], [500,50]:
        val = np.random.normal(0, 10)
        x = np.full(shape, val)        
        noisy = gaussian_noise(x, np.ones(shape[1]))
        assert((np.mean(noisy)-np.mean(x))**2<1.0)
        assert((np.std(noisy)-1.0)**2<0.1)
        noisy = gaussian_noise(x, np.full(shape[1],10.0))
        assert((np.std(noisy)-10.0)**2<0.1)
    

def test_cauchy_noise():
    np.random.seed(2012)
    for shape in [10,10], [100,1000], [500,50]:        
        val = np.random.normal(0, 10)
        x = np.full(shape, val)        
        noisy = cauchy_noise(x, np.ones(shape[1]))
        
def test_squared_error():
    for shape in [1, 1], [1, 10], [10, 1], [10, 10], [200, 10], [10, 200]:
        x = np.random.normal(0, 1, shape)
        y = np.random.normal(0, 1, shape)        
        assert np.allclose(squared_error(x, y, sigma=1), squared_error(x, y))
        assert np.all(squared_error(x, y, sigma=0.5) < squared_error(x, y))
        assert np.all(squared_error(x, y, sigma=2.0) > squared_error(x, y))
    

def test_heat_kernel():
    kernel = make_heat_adjusted(1.0)
    assert(kernel(0)==1.0)
    assert(kernel(1)<1.0)
    assert(kernel(1000)<1e-4)
    assert(np.allclose(kernel(3),np.exp(-3**2/2.0)))
    assert(kernel(-1)==kernel(1))
    assert(kernel(2)<kernel(1))
    a = np.zeros((10,10))
    b = np.ones((10,10))
    assert(np.all(kernel(a)==1.0))
    assert(np.all(kernel(b)<1.0))
    kernel_small = make_heat_adjusted(0.5)
    kernel_large = make_heat_adjusted(2.0)
    for k in -10, -5, -1, -0.5, 0.5, 1, 5, 10:
        assert(kernel_small(k)<kernel(k)<kernel_large(k))
    


