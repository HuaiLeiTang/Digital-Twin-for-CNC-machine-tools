# Data preprocessing of data
The data of acceleration, cutting force and acoustic emission can all be seen as vibration data. So data preprocessing of vibration is very usable and necessary. The data preprocessing contains **Unit transformation**, **Elimination of Trend Terms**,**Data denoising** and so on.

## Elimination of trend terms
Elimination of trend terms is based on Least Square Method. From the result figure, it can been seen that 1-4 order is mostly used in the elimination, because the vibration data is very large and the tred curve will be almost the same after 4 order.
<img src="https://github.com/1359246372/Digital-Twin-for-CNC-machine-tools/blob/master/data-driven/Data preprocessing/Trend eliminating with different order polynomials(1 to 9).png" width="100%">

Least Square Method realization can reference to this [link](https://blog.csdn.net/StupidAutofan/article/details/78924601#comments).

## Data denoising
The signal sampled by the sampler is often superimposed with noise signal, including power frequency signal, periodic interference signal and random interference signal, which makes the signal waveform with many burrs. In order to weaken the influence of interference signal and improve the smoothness of vibration curve, it is necessary to smooth the data. Moslty the smooth mehod is **Five-point sliding average smoothing**,**Five_three smoothing method**,**System function**

1. Five-point sliding average smoothing is a linear sliding average method,formula is like 
<img src="https://github.com/1359246372/Digital-Twin-for-CNC-machine-tools/blob/master/data-driven/Data preprocessing/Five-point sliding average smoothing formula.png">

The result after denoising is shown as below.
<img src="https://github.com/1359246372/Digital-Twin-for-CNC-machine-tools/blob/master/data-driven/Data preprocessing/Five-point sliding average smoothing method to denoise data.png">

2. Five_three smoothing method is a cubic polynomial smoothing method.   The fomula is like 
$$\[Y=a_{0}+a_{1}x+a_{2}x^{2}+a_{3}x^{2}\]$$
Take two adjacent points before and after each data point, and use cubic polynomial to approximate it. The coefficients a0, a1, a2, a3 are determined according to the least square principle, and it can be calculated by:
<img src="https://github.com/1359246372/Digital-Twin-for-CNC-machine-tools/blob/master/data-driven/Data preprocessing/Five_three smooth formula.png">
Five-point cubic smoothing filter can effectively remove the high-frequency random noise in the signal, and it is widely used in digital signal processing, and its filtering effect and flexibility are better than sliding average filter.

The comparation of these mehods is shown as below
<img src="https://github.com/1359246372/Digital-Twin-for-CNC-machine-tools/blob/master/data-driven/Data preprocessing/Smooth method comparation.png">
