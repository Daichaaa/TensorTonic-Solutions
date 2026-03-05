def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x_final = x0
    for i in range(steps):
        gradient = 2 * a * x_final + b
        x_final -= lr * gradient
        
    return x_final