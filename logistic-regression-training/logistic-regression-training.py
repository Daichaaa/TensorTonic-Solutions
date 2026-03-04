import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    n_samples, n_features  = X.shape
    
    w = np.zeros(n_features)
    b = 0.0

    for i in range(steps):
        z = np.dot(X,w) + b
        p = _sigmoid(z)
        loss = p - y
        gradient_w = np.dot(X.T, loss) / n_samples
        gradient_b = np.sum(loss) / n_samples
        w -= lr * gradient_w
        b -= lr * gradient_b
    return (w,b)