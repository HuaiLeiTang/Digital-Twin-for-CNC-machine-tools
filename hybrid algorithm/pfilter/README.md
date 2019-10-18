# pfilter
Basic Python particle filter. Plain SIR filtering, with systematic resampling. Written to be simple and clear; not necessarily most efficient or most flexible implementation. Depends on [NumPy](http://numpy.org) only. 

## Uses

This repo is useful for understanding how a particle filter works, or a quick way to develop a custom filter of your own from a relatively simple codebase. There are more mature and sophisticated packages for probabilistic filtering in Python (especially for Kalman filtering) if you want an off-the-shelf solution:

* [particles](https://github.com/nchopin/particles) Extensive particle filtering, including smoothing and quasi-SMC algorithms
* [FilterPy](https://github.com/rlabbe/filterpy) Provides extensive Kalman filtering and basic particle filtering.
* [pykalman](https://github.com/pykalman/pykalman) Easy to use KF, EKF and UKF implementations
* [simdkalman](https://github.com/oseiskar/simdkalman) Fast implmentations of plain Kalman filter banks.



## Installation

Available via PyPI:

    pip install pfilter
    
Or install the git version:

    pip install git+https://github.com/johnhw/pfilter.git

## Usage
Create a `ParticleFilter` object, then call `update(observation)` with an observation array to update the state of the particle filter.

Calling `update()` without an observation will update the model without any data, i.e. perform a prediction step only.

### Model

* Internal state space of `d` dimensions
* Observation space of `h` dimensions
* `n` particles estimating state in each time step

Particles are represented as an `(n,d)` matrix of states, one state per row. Observations are generated from this matrix into an `(n,h)` matrix of hypothesized observations via the observation function.

### Functions 
You need to specify at the minimum:

* an **observation function** `observe_fn(state) => observation matrix` which will return a predicted observation for an internal state.
* a function that samples from an **initial distributions** `prior_fn => state matrix` for all of the internal state variables. These are usually distributions from `scipy.stats`. The utility function `independent_sample` makes it easy to concatenate sampling functions to sample the whole state vector.
* a **weight function** `weight_fn(real_observed, hyp_observed) => weight vector` which specifies how well each of the `hyp_observed` arrays match the real observation `real_observed`. This must produce a strictly positive weight value for each hypothesized observation, where larger means more similar. This is often an RBF kernel or similar.

#### Missing observations
If you want to be able to deal with partial missing values in the observations, the weight function should support masked arrays. The `squared_error` function in `pfilter.py` does this, for example.


---

Typically, you would also specify:
*  **dynamics** a function `dynamics_fn(state) => predicted_state` to update the state based on internal (forward prediction) dynamics, and a 
* **diffusion** a function `noise_fn(predicted_state) => noisy_state` to add diffusion into the sampling process. 

---

You might also specify:

* **Internal weighting** a function `internal_weight_fn(state) => weight vector` which provides a weighting to apply on top of the weight function based on *internal* state. This is useful to impose penalties or to include learned inverse models in the inference.

## Attributes

The `ParticleFilter` object will have the following useful attributes after updating:

* `original_particles` the `(n,d)` collection of particles in the last update step
* `mean_state` the `(d,)` expectation of the state
* `mean_hypothesized`  the `(h,)` expectation of the hypothesized observations
* `cov_state` the `(d,d)` covariance matrix of the state
* `map_state` the `(d,)` most likely state
* `map_hypothesized` the `(h,)`  most likely hypothesized observation
* `weights` the  `(n,)` normalised weights of each particle


### Example

For example, assuming there is a function `blob` which draws a blob on an image of some size (the same size as the observation):

```python
        from pfilter import ParticleFilter, gaussian_noise, squared_error, independent_sample
        columns = ["x", "y", "radius", "dx", "dy"]
        from scipy.stats import norm, gamma, uniform 
        
        # prior sampling function for each variable
        # (assumes x and y are coordinates in the range 0-32)    
        prior_fn = independent_sample([uniform(loc=0, scale=32).rvs, 
                    uniform(loc=0, scale=32).rvs, 
                    gamma(a=2,loc=0,scale=10).rvs,
                    norm(loc=0, scale=0.5).rvs,
                    norm(loc=0, scale=0.5).rvs])
                                    
        # very simple linear dynamics: x += dx
        def velocity(x):
            xp = np.array(x)
            xp[0:2] += xp[3:5]        
        return xp
        
        # create the filter
        pf = pfilter.ParticleFilter(
                        prior_fn=prior_fn, 
                        observe_fn=blob,
                        n_particles=200,
                        dynamics_fn=velocity,
                        noise_fn=lambda x: 
                                    gaussian_noise(x, sigmas=[0.2, 0.2, 0.1, 0.05, 0.05]),
                        weight_fn=lambda x,y:squared_error(x, y, sigma=2),
                        resample_proportion=0.1,
                        column_names = columns)
                        
        # assuming image of the same dimensions/type as blob will produce
        pf.update(image) 
 ```

See the notebook [examples/example_filter.py](examples/test_filter.py) for a working example using `skimage` and `OpenCV` which tracks a moving white circle.
    
    
