import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.special import gamma, kv
import matplotlib.pyplot as plt


def tab1():

    st.header('Linear Kernel')
    st.write("""
    This kernel represents a linear relationship between input variables and is computed as the dot product of input vectors.
    """)

    # Define the function
    def z_function(x, y):
        return x * y

    # Generate x and y values
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = z_function(X, Y)

    # Create 3D surface plot
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])

    # Update layout
    fig.update_layout(scene=dict(xaxis_title='X',
                                yaxis_title='Y',
                                zaxis_title='Z'))

    # Streamlit display
    st.plotly_chart(fig)


def tab2():
    def generate_data(c, d, num_points):
        x = np.linspace(-10, 10, num_points)
        y = np.linspace(-10, 10, num_points)
        x, y = np.meshgrid(x, y)
        z = (x * y + c) ** d
        return x, y, z

    def main():
        st.header("Polynomial Kernel")
        st.write('''The polynomial kernel computes the similarity between two data points as the polynomial of their dot product.
                Polynomial kernels are particularly useful for capturing non-linear relationships in data.
                You can adjust the parameters 'c' and 'd' to explore different degrees and biases of polynomial features.''')

        # Define the function
        def z_function(x, y, c, d):
            return (x * y + c) ** d

        # Generate x and y values
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)

        # Choose parameters c and d
        c = st.slider("c", min_value=-10, max_value=10, value=0)
        d = st.slider("d", min_value=1, max_value=5, value=2)
        # Calculate Z values
        Z = z_function(X, Y, c, d)

        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])

        # Update layout
        fig.update_layout(scene=dict(xaxis_title='X',
                                    yaxis_title='Y',
                                    zaxis_title='Z'))

        # Streamlit display
        st.plotly_chart(fig)


    if __name__ == "__main__":
        main()


def tab3():
    def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
        sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return variance * np.exp(-0.5 / length_scale**2 * sqdist)

    def generate_surface(length_scale, variance):
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                Z[i, j] = rbf_kernel(np.array([[X[i, j], Y[i, j]]]), np.array([[0, 0]]), length_scale, variance)
        return X, Y, Z

    st.header('RBF Kernel')
    st.write("""
    This kernel, also known as the Gaussian kernel, exhibits smoothness and local stationarity properties. You can adjust the length scale and variance parameters to observe their effects on the kernel surface.
    """)

    length_scale = st.slider('Length Scale', min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    variance = st.slider('Variance', min_value=0.1, max_value=2.0, value=1.0, step=0.1)

    X, Y, Z = generate_surface(length_scale, variance)

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(title='Surface of RBF Kernel',
                    scene=dict(
                        xaxis=dict(title='X', range=[-5, 5]),
                        yaxis=dict(title='Y', range=[-5, 5]),
                        zaxis=dict(title='Kernel Value', range=[0, 1])),  # Adjust the range as needed
                    margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)


def tab4():
    def periodic_kernel(x1, x2, period=1.0, length_scale=1.0, variance=1.0):
        sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return variance * np.exp(-2 * np.sin(np.pi * np.sqrt(sqdist) / period)**2 / length_scale**2)

    def generate_surface_periodic(period, length_scale, variance):
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                Z[i, j] = periodic_kernel(np.array([[X[i, j], Y[i, j]]]), np.array([[0, 0]]), period, length_scale, variance)
        return X, Y, Z

    st.title('Periodic Kernel')
    st.write("""
    This kernel introduces periodicity into the Gaussian Process model, making it suitable for capturing periodic patterns in data. 
    You can adjust the period, length scale, and variance parameters to explore different periodic behaviours.
    """)

    period = st.slider('Period', min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    length_scale = st.slider('Length Scale', min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    variance = st.slider('Variance', min_value=0.1, max_value=2.0, value=1.0, step=0.1)

    X, Y, Z = generate_surface_periodic(period, length_scale, variance)

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(title='Surface of Periodic Kernel',
                    scene=dict(
                        xaxis=dict(title='X', range=[-5, 5]),
                        yaxis=dict(title='Y', range=[-5, 5]),
                        zaxis=dict(title='Kernel Value', range=[0, 2])),  # Adjust the range as needed
                    margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)

def tab5():
    def matern_kernel(x1, x2, nu=0.5, rho=1.0):
        r = np.linalg.norm(x1 - x2, axis=-1)
        term1 = (2**(1 - nu)) / gamma(nu)
        term2 = (np.sqrt(2 * nu) * r) / rho
        return term1 * (term2**nu) * kv(nu, term2)

    def generate_surface_matern(nu, rho):
        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = matern_kernel(np.dstack((X, Y)), np.array([[0, 0]]), nu, rho)
        return X, Y, Z

    st.header('Matérn Kernel')
    st.write("""
    This kernel offers a flexible family of covariance functions, characterised by smoothness parameter 'ν' and length scale 'ρ'. You can adjust these parameters to observe the impact on the kernel surface.
    """)

    nu = st.slider('ν (Smoothness)', min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    rho = st.slider('ρ (Length Scale)', min_value=0.1, max_value=5.0, value=1.0, step=0.1)

    X, Y, Z = generate_surface_matern(nu, rho)

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(title='Surface of Matérn Kernel',
                    scene=dict(
                        xaxis=dict(title='X', range=[-1, 1]),
                        yaxis=dict(title='Y', range=[-1, 1]),
                        zaxis=dict(title='Kernel Value', range=[np.min(Z), np.max(Z)])),
                    margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)

def tab6():
    def spectral_mixture_kernel(x1, x2, weights, means, variances):
        num_mixtures = len(weights)
        k = 0
        for i in range(num_mixtures):
            k += weights[i] * np.exp(-2 * np.pi**2 * variances[i] * (x1 - x2)**2) * np.cos(2 * np.pi * means[i] * (x1 - x2))
        return k

    def generate_surface_spectral_mixture(weights, means, variances):
        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = spectral_mixture_kernel(X[i, j], Y[i, j], weights, means, variances)
        return X, Y, Z

    st.title('Spectral Mixture')
    st.write("""
    This kernel is a linear combination of periodic components, offering flexibility in capturing complex patterns in data. You can specify the number of mixture components and adjust their weights, means, and variances to customise the kernel surface.
    """)

    num_mixtures = st.slider('Number of Mixtures', min_value=1, max_value=5, value=2, step=1)

    weights = []
    means = []
    variances = []

    for i in range(num_mixtures):
        weight = st.slider(f'Weight {i+1}', min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        weights.append(weight)
        mean = st.slider(f'Mean {i+1}', min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        means.append(mean)
        variance = st.slider(f'Variance {i+1}', min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        variances.append(variance)

    X, Y, Z = generate_surface_spectral_mixture(weights, means, variances)

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(title='Surface of Spectral Mixture Kernel',
                    scene=dict(
                        xaxis=dict(title='X', range=[-1, 1]),
                        yaxis=dict(title='Y', range=[-1, 1]),
                        zaxis=dict(title='Kernel Value')),
                    margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("Kernels as 3D surfaces Visualisation")

    with st.sidebar:
        # Create tabs
        st.caption('''This Streamlit app allows you to visualise various kernels used in Gaussian Processes. 
                   Gaussian Processes are a powerful tool in machine learning for regression and probabilistic classification. 
                   Kernels play a crucial role in defining the covariance function of Gaussian Processes, influencing the smoothness, periodicity, and other characteristics of the inferred functions. 
                   This app provides interactive visualiaation of different kernel functions, enabling users to explore their properties and understand their impact on Gaussian Process models.''')
       
        tabs = ["Linear", "Polynomial","RBF","Periodic","Matérn","Spectral Mixture"]
        selected_tab = st.radio("Select a kernel", tabs)

        #Ana's signature
        st.divider()
        st.caption('<p style="text-align:center">made with ♡ by Ana</p>', unsafe_allow_html=True)


    # Display the selected tab
    if selected_tab == "Linear":
        tab1()
    elif selected_tab == "Polynomial":
        tab2()
    elif selected_tab == "RBF":
        tab3()
    elif selected_tab == "Periodic":
        tab4()
    elif selected_tab == "Matérn":
        tab5()
    elif selected_tab == "Spectral Mixture":
        tab6()

if __name__ == "__main__":
    main()
