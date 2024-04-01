import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


# Example sine function for dense signal
# Generate dense signal using a simple sine function
dense_time_1 = np.linspace(0, 10, 100)  # Time points from 0 to 10
dense_time_ = np.linspace(0, 20, 200)  # Time points from 0 to 10
dense_signal_ = np.sin(dense_time_1)

# Define the range for sparse time
sparse_time_range = (0, 20)

# Filter dense time points within the range
filtered_indices = (dense_time_ >= sparse_time_range[0]) & (dense_time_ <= sparse_time_range[1])
filtered_dense_time = dense_time_[filtered_indices]
filtered_dense_signal = dense_signal_[filtered_indices]

# Number of data points to be selected for the sparse signal
num_sparse_points = 100

# Random Randomly select indices from the filtered dense signal
sparse_indices = np.random.choice(filtered_dense_time.shape[0], size=num_sparse_points, replace=False)
# Sort the selected indices
sparse_indices = np.sort(sparse_indices)

# Generate sparse signal using the randomly selected indices
sparse_time_ = filtered_dense_time[sparse_indices]
sparse_signal_ = filtered_dense_signal[sparse_indices]

# Moving
# Calculate the time shift
time_shift = 5
# Update each time point in sparse_time_
sparse_time_ -= time_shift
# Move the sparse signal one point downwards
sparse_signal_ -= 1

# Scaling
# Scale down the sparse signal to half the amplitude of the dense signal
sparse_signal_ /= 2
# Scale up the sparse signal to twice the amplitude of the dense signal
#sparse_signal_ *= 2

# Add noise to the sparse signal
#noise = np.random.normal(0, 0.1, size=sparse_signal_.shape)
#sparse_signal_ += noise

# Convert NumPy arrays to lists
dense_time = dense_time_1.tolist()
dense_signal = dense_signal_.tolist()
sparse_time = sparse_time_.tolist()
sparse_signal = sparse_signal_.tolist()

# Multiply sparse time and dense time by 10
dense_time = [t * 10 for t in dense_time]
sparse_time = [t * 10 for t in sparse_time]



# First plot (dense signal/sparse signal)
# Plot the dense signal
plt.scatter(dense_time, dense_signal, label='Dense Signal (SST)', color='blue')

# Plot the sparse signal
plt.scatter(sparse_time, sparse_signal, label='Sparse Signal (SPP)', color='red')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series Data Before Interpolation')
plt.legend()

# Show plot
plt.grid(True)
plt.show()


# Interpolation
# Define the range for interpolation
interpolation_range = (sparse_time[0], sparse_time[-1])

# Generate interpolated sparse time within the specified range
interpolated_sparse_time = [time for time in dense_time if interpolation_range[0] <= time <= interpolation_range[1]]

# Sparse signal interpolation function
def interpolate_sparse_signal(interpolated_sparse_time, sparse_time, sparse_signal):
    # Train Gaussian process regression model
    kernel = 1.0 * Matern(length_scale=1.0)  # Matern kernel
    gp_model = GaussianProcessRegressor(kernel=kernel, alpha = 0.0, n_restarts_optimizer=10)
    gp_model.fit(np.array(sparse_time).reshape(-1, 1), sparse_signal)
    
    # Predict sparse signal values at dense time points
    interpolated_sparse_signal, sigma = gp_model.predict(np.array(interpolated_sparse_time).reshape(-1, 1), return_std=True)
    
    # Calculate the distance between each interpolated time point and the nearest sparse time point
    distances = np.abs(np.subtract.outer(interpolated_sparse_time, sparse_time)).min(axis=1)
    sigma += 0.005 * (1 + distances)  # Adjust sigma based on distance
    
    return interpolated_sparse_signal, sigma

# Interpolate sparse signal using the interpolated sparse time
interpolated_sparse_signal, sigma = interpolate_sparse_signal(interpolated_sparse_time, sparse_time, sparse_signal)



# Check print
# Print the values side by side
print("Sparse Time  |  Sparse Signal ")
for sparse_t, sparse_s in zip(sparse_time, sparse_signal):
    print(f"{sparse_t:.2f}          |    {sparse_s:.4f}")



# Second plot (sigma/interpolated sparse signal)
# Plot the dense signal
plt.scatter(dense_time, dense_signal, label='Dense Signal (SST)', color='blue')

# Plot the sparse signal
plt.scatter(sparse_time, sparse_signal, label='Sparse Signal (SPP)', color='red')

# Plot the interpolated sparse signal
plt.plot(interpolated_sparse_time, interpolated_sparse_signal, label='Interpolated Sparse Signal', color='green')

# Plot uncertainty represented by sigma with adjusted transparency
plt.fill_between(interpolated_sparse_time, interpolated_sparse_signal - sigma, interpolated_sparse_signal + sigma, color='gray', alpha=0.2, label='Uncertainty (Sigma)')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series Data with Interpolated Sparse Signal and Uncertainty')
plt.legend()

# Show plot
plt.grid(True)
plt.show()



# Crosscorrelation
# Lag max

lag_max = 40

# Print variables before the function call
print("Before Function Call:")
print("Interpolated Sparse Signal | Dense Signal | Dense Time | Sigma | Lag Max")
for i in range(len(interpolated_sparse_signal)):
    print(f"{interpolated_sparse_signal[i]:.4f} | {dense_signal[i]:.4f} | {dense_time[i]:.4f} | {sigma[i]:.4f} | {lag_max}")

def cross_sparsedense(interpolated_sparse_signal, interpolated_sparse_time, dense_signal, dense_time, lag_max, sigma):
    l1 = len(dense_time)
    l2 = len(interpolated_sparse_time)
    correlation = np.zeros(2 * lag_max + 1)
    lag_values = np.arange(-lag_max, lag_max + 1)
    correlation_with_lag = {}  # Dictionary to store correlation values with lag
    
    # Find the range of valid shifts based on the minimum and maximum time values in the interpolated sparse signal
    min_interpolated_time = min(interpolated_sparse_time)
    max_interpolated_time = max(interpolated_sparse_time)
    min_shift = max(-lag_max, min_interpolated_time - dense_time[0])
    max_shift = min(lag_max, max_interpolated_time - dense_time[-1])
    
    for lag in lag_values:
        time2x = np.array(interpolated_sparse_time) - lag
        time1_time2 = np.zeros(l2, dtype=int)
        
        # Shift the interpolated sparse signal within the valid range of shifts
        shifted_time2x = np.clip(time2x, dense_time[0] + min_shift, dense_time[-1] + max_shift)
        time1_time2 = np.round(shifted_time2x - dense_time[0]).astype(int)
        
        s = 0
        total_weight = 0
        for j in range(l2):
            if time1_time2[j] < 0 or time1_time2[j] >= l1:
                continue
            if sigma[j] == 0:
                weight = 0
            else:
                weight = 1
            s += dense_signal[time1_time2[j]] * interpolated_sparse_signal[j] * weight
            total_weight += weight
        
        if total_weight > 0:
            correlation[-lag + lag_max] = s / total_weight
            correlation_with_lag[-lag] = s / total_weight
        else:
            correlation[-lag + lag_max] = 0
            correlation_with_lag[-lag] = 0
    
    return correlation, lag_values, correlation_with_lag

# Example usage
correlation, lag_values, correlation_with_lag = cross_sparsedense(interpolated_sparse_signal, interpolated_sparse_time, dense_signal, dense_time, lag_max, sigma)

# Print variables after the function call
print("\nAfter Function Call:")
print("Correlation Array | Correlation With Lag Dictionary")
for lag, corr_value in correlation_with_lag.items():
    print(f"{correlation[lag + lag_max]:.4f} | {lag}: {corr_value:.4f}")


# Check for the correlation values calculated 
# Print correlation values with lag
for lag, corr_value in correlation_with_lag.items():
    print(f"Lag: {lag}, Correlation: {corr_value}")

# Find the maximum correlation value (ignoring NaN values)
max_correlation_value = np.nanmax(correlation)

# Find the index of the maximum correlation value (ignoring NaN values)
max_corr_index = np.nanargmax(correlation)

# Find the lag corresponding to the maximum correlation value
max_corr_lag = lag_values[max_corr_index]

# Now, check if there are multiple maximum correlation values
if np.sum(np.isnan(correlation)) < len(correlation):  # Check if there are NaN values
    if len(np.unique(correlation)) > 1:  # Check if there are multiple unique correlation values
        # If there are multiple maximum correlation values, choose the one with the smallest lag value
        if len(np.unique(correlation)) > 1:
            max_corr_lags = lag_values[np.where(correlation == max_correlation_value)[0]]
            if len(max_corr_lags) > 1:
                max_corr_lag = min(max_corr_lags, key=abs)
else:
    max_corr_lag = 0  # If all correlation values are NaN, set max_corr_lag to 0



# Third plot lags and correlation
# Shift the sparse signal by the suggested lag
suggested_lag = max_corr_lag
shifted_sparse_time = [t + suggested_lag for t in sparse_time]

# Print the maximum correlation value and corresponding lag
print(f"Max Correlation: {max_correlation_value:.2f} at Lag: {max_corr_lag}")

# Plot the dense signal
plt.plot(dense_time, dense_signal, label='Dense Signal')

# Plot the sparse signal
plt.scatter(sparse_time, sparse_signal, color='red', label='Sparse Signal')

# Plot the interpolated sparse signal
plt.scatter(interpolated_sparse_time, interpolated_sparse_signal, color='slateblue', alpha=0.5, label='Interpolated Sparse Signal')

# Plot the sigma
plt.scatter(interpolated_sparse_time, sigma, color='gold', label='sigma')

# Plot the shifted sparse signal with the suggested lag
plt.scatter(shifted_sparse_time, sparse_signal, color='green', alpha=0.5, label=f'Sparse Signal (Shifted by Lag {suggested_lag})')

# Add labels and legend
plt.xlabel('Time')
plt.ylabel('Signal Value')
plt.title('Dense and Sparse Signals with Suggested Lag')

# Add maximum correlation value with its respective lag as label
plt.text(0.95, 0.95, f'Max Correlation: {max_correlation_value:.2f} at Lag: {max_corr_lag}', ha='right', va='top', transform=plt.gca().transAxes, fontsize=10)

plt.legend()

# Show plot
plt.grid(True)
plt.show()