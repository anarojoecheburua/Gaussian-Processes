import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D


# Streamlit app
st.title('Gaussian Processes - The Basics 🪄')
st.subheader('Welcome to the wonderful world of Gaussian Processes!')
st.write('This app allows you to explore Gaussian Processes and visualise their behaviour.')


#Tabs
tab1, tab2, tab3 = st.tabs(["Gaussian Distribution", "Parametric vs Non-parametric", "Distributions Over Functions"])

#Tab 1 - Normal Distribution

with tab1:

    def gaussian_distribution(mean, std_dev):
        x = np.linspace(-10, 10, 1000)
        y = norm.pdf(x, loc=mean, scale=std_dev)
        return x, y

    st.title("Understanding a Gaussian Distribution")

    st.write("""
    This section of the app allows you to explore and visualise the Gaussian distribution, also known as the normal distribution. 
    The Gaussian distribution is a continuous probability distribution that is symmetric around its mean.
    """)

    st.markdown("""
    The Gaussian distribution is crucial in understanding Gaussian processes - this distribution has a well-defined probability
    density function that represents the likelihood of different outcomes. Gaussian processes often rely on this probability density function
    to model uncertainty and variability in data.
    """)

    st.caption("""
    → Adjust the sliders in the sidebar to change the mean and standard deviation of the Gaussian distribution. 
    The plot will dynamically update to show how these parameters affect the shape of the distribution.
    """)

    # Initial values
    default_mean = 0.0
    default_std_dev = 1.0

    # Sliders for adjusting mean and standard deviation
    st.sidebar.title('Gaussian Distribution')
    mean = st.sidebar.slider("Mean", min_value=-10.0, max_value=10.0, value=default_mean)
    std_dev = st.sidebar.slider("Standard Deviation", min_value=0.5, max_value=10.0, value=default_std_dev, step=0.1)
    st.sidebar.divider()

    # Generate Gaussian distribution based on sliders
    x, y = gaussian_distribution(mean, std_dev)

    # Plot the Gaussian distribution
    fig, ax = plt.subplots()
    ax.plot(x, y, label=f'Mean={mean}, Std Dev={std_dev}')
    ax.set_title('Gaussian Distribution')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Probability Density Function')
    ax.set_ylim([0, 0.6])  # Set y-axis limits from 0 to 0.6
    ax.legend()

    # Show the plot using Streamlit
    st.pyplot(fig)


#Tab 2 - Parametric vs Non-Parametric
    
with tab2:
    # Set initial axis limits
    line_xlim = (-10, 10)
    line_ylim = (-20, 20)
    rbf_xlim = (-5, 5)
    rbf_ylim = (-5, 5)
    rbf_zlim = (0, 10)

    # Function to generate line plot
    def generate_line_plot(ax, slope, y_intercept):
        ax.clear()
        x = np.linspace(-10, 10, 100)
        y = slope * x + y_intercept
        ax.plot(x, y)
        ax.set_title('Line Plot')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_xlim(line_xlim)
        ax.set_ylim(line_ylim)

    # Function to generate 2D surface plot of RBF kernel
    def generate_rbf_surface(figure, variance, lengthscale):
        ax = figure.add_subplot(122, projection='3d')  # Use add_subplot on the figure
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        x, y = np.meshgrid(x, y)
        rbf_kernel = variance * np.exp(-(x**2 + y**2) / (2 * lengthscale**2))
        ax.plot_surface(x, y, rbf_kernel, cmap='viridis')
        ax.set_title('RBF Kernel')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Kernel Value')
        ax.set_xlim(rbf_xlim)
        ax.set_ylim(rbf_ylim)
        ax.set_zlim(rbf_zlim)

    # Streamlit app
    st.title('Parametric vs non-parametric')

    # Explanation text
    st.write("""
        This part of the app allow you to explore and compare a linear function and a Radial Basis Function (RBF) kernel.""")
    
    st.write("""
        A parametric form refers to a specific formula with a fixed set of parameters. 
        These parameters determine the shape, behaviour, and characteristics of the function.
        Non-parametric models, on the other hand, do not have a fixed functional form with a predetermined number of parameters""")
    
    st.write("""Gaussian Processes fall into the category of non-parametric models. 
        That means it doesn't have a fixed number of parameters like some other models. However, it does use parameters to describe its kernel""")
    
    st.caption("""
        → Adjust the sliders on the sidebar to modify the parameters of each model.""")
    
    # Sidebar with sliders
    st.sidebar.title('Parametric vs Non-parametric')
    st.sidebar.caption("Line")
    slope = st.sidebar.slider('Slope', min_value=-10.0, max_value=10.0, value=1.0, step=0.1)
    y_intercept = st.sidebar.slider('Y Intercept', min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
    st.sidebar.latex(f'y = {slope}x + {y_intercept}')
    st.sidebar.caption("Kernel")
    variance = st.sidebar.slider('Variance', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    lengthscale = st.sidebar.slider('Lengthscale', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    st.sidebar.latex(f'K(x, y) = {variance} * exp(-((x^2 + y^2) / (2 * {lengthscale}^2)))')
    st.sidebar.divider()

    # Line and RBF kernel plots side by side
    fig = plt.figure(figsize=(12, 5))

    # Create the first subplot for the line plot
    line_ax = fig.add_subplot(121)
    generate_line_plot(line_ax, slope, y_intercept)

    # Create the second subplot for the RBF kernel plot
    generate_rbf_surface(fig, variance, lengthscale)

    st.pyplot(fig)


#Tab 3 - Distributions over functions

with tab3:

    def kernel(a, b, param):
        sqdist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2*np.dot(a, b.T)
        epsilon = 1e-6  # Small epsilon value to ensure matrix invertibility
        return np.exp(-0.5 * (1/(param + epsilon)) * sqdist)

    def save_figure_to_bytes(figure):
        buf = BytesIO()
        figure.savefig(buf, format="png")
        buf.seek(0)
        return buf

    # Function to generate samples from the GP prior
    def gp_prior_samples(Xtest, param, num_samples):
        K_ss = kernel(Xtest, Xtest, param)
        L = np.linalg.cholesky(K_ss + 1e-6*np.eye(len(Xtest)))  # Add jitter to ensure positive definiteness
        f_prior = np.dot(L, np.random.normal(size=(len(Xtest), num_samples)))
        return f_prior


    # Function to generate samples from the GP posterior
    def gp_posterior_samples(Xtrain, ytrain, Xtest, param, num_samples):
        K = kernel(Xtrain, Xtrain, param)
        L = np.linalg.cholesky(K + 1e-6*np.eye(len(Xtrain)))  # Add jitter to ensure positive definiteness
        K_s = kernel(Xtrain, Xtest, param)
        Lk = np.linalg.solve(L, K_s)
        mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((len(Xtest),))
        s2 = np.diag(kernel(Xtest, Xtest, param)) - np.sum(Lk**2, axis=0)
        stdv = np.sqrt(np.maximum(s2, 0))  # Ensure stdv is non-negative
        L = np.linalg.cholesky(kernel(Xtest, Xtest, param) + 1e-6*np.eye(len(Xtest)) - np.dot(Lk.T, Lk))
        f_post = mu.reshape(-1, 1) + np.dot(L, np.random.normal(size=(len(Xtest), num_samples)))
        return mu, stdv, f_post

    #Header and explanation
    st.header('Distributions Over Functions')
    st.write("""
        A way of understanding a Gaussian Process is as a distribution over functions, where each function can be thought of as an infinite-dimensional vector""")
    
    st.write("""
        Instead of giving a single prediction, they provide a distribution of possible outcomes.""")
    
    st.caption(' → Adjust the "Kernel Parameter" slider in the sidebar to control the width of the Gaussian distribution around each point in the GP. Smaller values result in localised effects, while larger values lead to smoother, more global patterns.')

    st.caption(' →Use the "Number of Samples" slider in the sidebar to set the number of function samples to display. More samples give you a better understanding of the range of possible functions generated by the GP.')
  
    st.caption(' →Modify the "Number of Data Points" slider to change the quantity of training data points used to fit the GP. More data points make the GP more sensitive to local variations in the data.')

    # Sidebar for parameter selection
    st.sidebar.title('Disributions over functions')
    param = st.sidebar.slider('Kernel Parameter', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    num_samples = st.sidebar.slider('Number of Samples', min_value=1, max_value=10, value=3, step=1)
    num_data_points = st.sidebar.slider('Number of Data Points', min_value=0, max_value=20, value=5, step=1)
    
    # Test data
    n = 50
    Xtest = np.linspace(-5, 5, n).reshape(-1, 1)

    # Generate and display samples from the GP prior
    st.subheader('Gaussian Process Prior')
    st.write('Explore the "Gaussian Process Prior" section to visualize samples drawn from the GP prior. This represents the distribution of functions before observing any data.')
    f_prior = gp_prior_samples(Xtest, param, num_samples)
    fig_prior, ax_prior = plt.subplots()
    ax_prior.plot(Xtest, f_prior)
    st.image(save_figure_to_bytes(fig_prior))

    # Noiseless training data
    Xtrain = np.linspace(-4, 4, num_data_points).reshape(-1, 1)
    ytrain = np.sin(Xtrain)

    # Generate and display samples from the GP posterior
    st.subheader('Gaussian Process Posterior')
    st.write('In the "Gaussian Process Posterior" section, observe how the GP adapts to noisy training data. The black circles represent training data points, and the red dashed line represents the posterior mean.')
    mu, stdv, f_post = gp_posterior_samples(Xtrain, ytrain, Xtest, param, num_samples)
    fig_post, ax_post = plt.subplots()
    ax_post.plot(Xtrain, ytrain, 'ko', ms=8, label='Training Data')
    ax_post.plot(Xtest, f_post)
    ax_post.fill_between(Xtest.flat, mu-2*stdv, mu+2*stdv, color="#dddddd")
    ax_post.plot(Xtest, mu, 'r--', lw=2, label='Posterior Mean')
    ax_post.legend()
    ax_post.axis([-5, 5, -3, 3])
    st.image(save_figure_to_bytes(fig_post))

    #Ana's signature
    st.sidebar.divider()

    st.sidebar.caption('<p style="text-align:center">made with ♡ by Ana</p>', unsafe_allow_html=True)
   







