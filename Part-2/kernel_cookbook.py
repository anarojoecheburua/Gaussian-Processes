import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define kernels
def linear_kernel(x1, x2, variance=1.0, bias=0.0):
    return variance * np.dot(x1, x2) + bias

def polynomial_kernel(x1, x2, degree=2, variance=1.0, bias=0.0):
    return (variance * np.dot(x1, x2) + bias) ** degree

def rbf_kernel(x1, x2, lengthscale=1.0, variance=1.0):
    return variance * np.exp(-0.5 * np.sum((x1 - x2) ** 2) / lengthscale ** 2)

def periodic_kernel(x1, x2, lengthscale=1.0, period=1.0, variance=1.0):
    return variance * np.exp(-2 * (np.sin(np.pi / period * np.abs(x1 - x2)) ** 2) / lengthscale ** 2)

def matern_kernel(x1, x2, lengthscale=1.0, variance=1.0, nu=1.5):
    dist = np.abs(x1 - x2)
    sqrt_3nu = np.sqrt(3 * nu)
    arg = sqrt_3nu * dist / lengthscale
    return variance * (1 + arg) * np.exp(-arg)

def spectral_mixture_kernel(x1, x2, weights, lengthscales):
    return np.sum([weights[i] * np.exp(-0.5 * np.sum((x1 - x2) ** 2) / lengthscales[i] ** 2) for i in range(len(weights))])

# Sample functions from prior
def sample_functions_prior(kernel, params, num_samples=5, x_min=-5, x_max=5, num_points=100):
    x = np.linspace(x_min, x_max, num_points)
    K = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            K[i, j] = kernel(x[i], x[j], **params)
    mean = np.zeros(num_points)
    samples = np.random.multivariate_normal(mean, K, num_samples)
    return x, samples, mean, np.diag(K)

# Compute posterior
def compute_posterior(x_train, y_train, kernel, params, x_test):
    K_train = np.zeros((len(x_train), len(x_train)))
    for i in range(len(x_train)):
        for j in range(len(x_train)):
            K_train[i, j] = kernel(x_train[i], x_train[j], **params)
    K_train += 1e-8 * np.eye(len(x_train))
    
    K_test_train = np.zeros((len(x_test), len(x_train)))
    for i in range(len(x_test)):
        for j in range(len(x_train)):
            K_test_train[i, j] = kernel(x_test[i], x_train[j], **params)
    
    K_test = np.zeros((len(x_test), len(x_test)))
    for i in range(len(x_test)):
        for j in range(len(x_test)):
            K_test[i, j] = kernel(x_test[i], x_test[j], **params)
    K_test += 1e-8 * np.eye(len(x_test))
    
    K_train_inv = np.linalg.inv(K_train)
    mean = np.dot(np.dot(K_test_train, K_train_inv), y_train)
    cov = K_test - np.dot(np.dot(K_test_train, K_train_inv), K_test_train.T)
    return mean, cov

# Streamlit app
def main():
    st.title('Kernels Explorer 🔎')

    st.write('''This app illustrates the prior and posterior of a Gaussian Process Regressor with different kernels. Mean, uncertainty, and 5 samples are shown for both prior and posterior distributions.''')
    st.caption('''Kernels play a very important role in determining the characteristics of both the prior and posterior distributions of the Gaussian Process. 
            They encapsulate the underlying assumptions about the function being modeled by specifying the "similarity" between pairs of data points.
            Kernels define the degree of covariance between different inputs, thereby shaping the behavior and predictions of the Gaussian Process model.''')
    
    st.divider()

    kernel_options = ['Linear', 'Polynomial', 'RBF', 'Periodic', 'Matern', 'Spectral Mixture']
    kernel_choice = st.selectbox('Choose Kernel', kernel_options)

    if kernel_choice == 'Linear':
        st.sidebar.header('Linear Kernel Hyperparameters')
        variance = st.sidebar.slider('Variance', min_value=0.1, max_value=10.0, value=1.0)
        bias = st.sidebar.slider('Bias', min_value=-5.0, max_value=5.0, value=0.0)
        params = {'variance': variance, 'bias': bias}
        kernel = linear_kernel
    elif kernel_choice == 'Polynomial':
        st.sidebar.header('Polynomial Kernel Hyperparameters')
        degree = st.sidebar.slider('Degree', min_value=1, max_value=5, value=2)
        variance = st.sidebar.slider('Variance', min_value=0.1, max_value=10.0, value=1.0)
        bias = st.sidebar.slider('Bias', min_value=-5.0, max_value=5.0, value=0.0)
        params = {'degree': degree, 'variance': variance, 'bias': bias}
        kernel = polynomial_kernel
    elif kernel_choice == 'RBF':
        st.sidebar.header('RBF Kernel Hyperparameters')
        lengthscale = st.sidebar.slider('Lengthscale', min_value=0.1, max_value=5.0, value=1.0)
        variance = st.sidebar.slider('Variance', min_value=0.1, max_value=10.0, value=1.0)
        params = {'lengthscale': lengthscale, 'variance': variance}
        kernel = rbf_kernel
    elif kernel_choice == 'Periodic':
        st.sidebar.header('Periodic Kernel Hyperparameters')
        lengthscale = st.sidebar.slider('Lengthscale', min_value=0.1, max_value=5.0, value=1.0)
        period = st.sidebar.slider('Period', min_value=0.1, max_value=5.0, value=1.0)
        variance = st.sidebar.slider('Variance', min_value=0.1, max_value=10.0, value=1.0)
        params = {'lengthscale': lengthscale, 'period': period, 'variance': variance}
        kernel = periodic_kernel
    elif kernel_choice == 'Matern':
        st.sidebar.header('Matern Kernel Hyperparameters')
        lengthscale = st.sidebar.slider('Lengthscale', min_value=0.1, max_value=5.0, value=1.0)
        variance = st.sidebar.slider('Variance', min_value=0.1, max_value=10.0, value=1.0)
        nu = st.sidebar.select_slider('ν', options=[0.5, 1.5, 2.5])
        params = {'lengthscale': lengthscale, 'variance': variance, 'nu': nu}
        kernel = matern_kernel
    elif kernel_choice == 'Spectral Mixture':
        st.sidebar.header('Spectral Mixture Kernel Hyperparameters')
        num_components = st.sidebar.slider('Number of Components', min_value=1, max_value=5, value=2)
        weights = [st.sidebar.slider(f'Weight {i+1}', min_value=0.1, max_value=2.0, value=1.0) for i in range(num_components)]
        lengthscales = [st.sidebar.slider(f'Lengthscale {i+1}', min_value=0.1, max_value=5.0, value=1.0) for i in range(num_components)]
        params = {'weights': weights, 'lengthscales': lengthscales}
        kernel = spectral_mixture_kernel

    st.header('Prior Samples')
    num_samples_prior = 5
    x_prior, samples_prior, mean_prior, diag_cov_prior = sample_functions_prior(kernel, params, num_samples=num_samples_prior)
    for i in range(num_samples_prior):
        plt.plot(x_prior, samples_prior[i], alpha=0.5, label=f'Sample {i+1}')
    plt.plot(x_prior, mean_prior, color='black', linestyle='--', label='Mean')
    plt.fill_between(x_prior, mean_prior - 2 * np.sqrt(diag_cov_prior), mean_prior + 2 * np.sqrt(diag_cov_prior), color='blue', alpha=0.2, label='Uncertainty')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    st.pyplot(plt)

    st.write('''Sample functions drawn from the prior distribution of a Gaussian Process. Note that each kernel function imposes a unique structural bias on the distribution of functions.''')

    st.header('Posterior Samples')

    x_train = st.text_input('Input X vector (comma-separated)', '-2,-1,0,1,2')
    y_train = st.text_input('Input Y vector (comma-separated)', '0,1,0,1,0')

    try:
        x_train = np.array([float(x) for x in x_train.split(',')])
        y_train = np.array([float(y) for y in y_train.split(',')])
    except:
        st.error('Invalid input. Please enter comma-separated numerical values.')
        return

    mean_posterior, cov_posterior = compute_posterior(x_train, y_train, kernel, params, x_prior)
    num_samples_posterior = 5
    samples_posterior = np.random.multivariate_normal(mean_posterior, cov_posterior, num_samples_posterior)

    plt.figure()  # Clear the current figure
    for i in range(num_samples_posterior):
        plt.plot(x_prior, samples_posterior[i], alpha=0.5, label=f'Sample {i+1}')
    plt.scatter(x_train, y_train, color='red', label='Observations')
    plt.plot(x_prior, mean_posterior, color='black', label='Mean')
    plt.fill_between(x_prior, mean_posterior - 2 * np.sqrt(np.diag(cov_posterior)),
                     mean_posterior + 2 * np.sqrt(np.diag(cov_posterior)),
                     color='blue', alpha=0.2, label='Uncertainty')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    st.pyplot(plt)

    st.write('''Sample functions drawn from the posterior distribution of a Gaussian Procsess. Red points are observation data.
             See how Gaussian Processes explicitly model the uncertainty of the predictive function. ''')
    
    #Ana's signature
    st.sidebar.divider()
    st.sidebar.caption('<p style="text-align:center">made with ♡ by Ana</p>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()






















