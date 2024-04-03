# sparse-dense-crosscorrelation

## Cross-correlation between Sparse and Dense Signals
This Python code aims to determine the optimal correlation between two signals—one sparse and one dense—by aligning them with varying time lags and calculating the cross-correlation coefficient.

## Motivation
Traditional cross-correlation methods are typically designed for signals of the same nature, either sparse or dense. However, in scenarios where one signal is sparse and the other dense, existing methods are inadequate. This code addresses this gap by introducing a novel approach to compute cross-correlation between sparse and dense signals.

## Functionality Overview
1- Signal Visualization: The code begins by plotting the two signals—sparse and dense—providing a visual representation of the data.

2- Sparse Signal Interpolation: Utilizing Gaussian process regression with a Matern kernel, the sparse signal is interpolated to match the density of the dense signal. The interpolation process adjusts for signal irregularities and incorporates uncertainty (sigma) based on distance and measurement errors.

3- Cross-correlation Calculation: The core functionality involves computing the cross-correlation between the interpolated sparse signal and the dense signal. This step involves shifting one signal relative to the other across a specified range of time lags and calculating the correlation coefficient.

4- Lag Selection: After determining the cross-correlation coefficients, the code identifies the maximum correlation value and its corresponding lag, signifying the optimal alignment between the two signals.

5- Visualization of Aligned Signals: Finally, the shifted sparse signal, aligned with the dense signal according to the selected lag, is plotted alongside the original dense signal, providing a clear depiction of their correlation.

## Key Contributions
- Novel approach to computing cross-correlation between sparse and dense signals.
- Integration of Gaussian process regression for signal interpolation, accounting for uncertainty and measurement errors.
- Automated selection of optimal alignment between signals, facilitating accurate correlation analysis.

## Usage
To utilize this code:
- Input the sparse and dense signal data.
- Specify the maximum lag for alignment.
- Run the script to visualize the signals, calculate cross-correlation, and identify optimal alignment.

## Dependencies
- matplotlib for data visualization.
- numpy for numerical computations.
- scikit-learn for Gaussian process regression.

## Example Explanation
In this example, we generate a sine function (dense signal) and create a sparse signal by selecting a few points from the dense signal. The purpose of this example is to demonstrate the functionality of the interpolation and cross-correlation functions implemented in the code.
### Purpose of the Example
The generated sparse signal is initially identical (or barely shifted) to the original position of the sparse signal. Ideally, when calculating the correlation with lags, the biggest correlation value should be around lag 0, as the dense signal and the sparse signal are seamlessly the same (one is sparse, one is dense). However, we shift and scale the sparse signal to assess if the interpolation function and the cross-correlation function work correctly. 
### Experimentation
To experiment with the code, one can modify the values of the shifting and scaling of the sparse signal. By doing so, we can observe if the interpolation and cross-correlation calculations, as well as the visualization of the shifted sparse signal, are performed correctly.
### Visual Assessment
The final plot visually assesses if the maximum correlation value and corresponding lag are correct. In an accurate scenario, the peaks of the signals should match, indicating that the interpolation and cross-correlation functions have been implemented effectively.
### Example usage of the cross-correlation function
correlation, lag_values, correlation_with_lag = cross_sparsedense(interpolated_sparse_signal, interpolated_sparse_time, dense_signal, dense_time, lag_max, sigma)

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgements
This project was inspired by the need for robust cross-correlation methods in scenarios involving signals of different densities.

## Contributors
- Ayman (https://github.com/Aymansvvd)
- Alex (https://github.com/Alexbernardino) 
