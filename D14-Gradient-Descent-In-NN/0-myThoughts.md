# Batch GD vs Stochastic GD (MSE)

## 1. Batch Gradient Descent (BGD)

- Uses the **entire training dataset** to calculate the gradient.
- Updates weights only once after processing all training samples.

### Pseudocode

```python
Initialize weights theta
alpha = 0.01

for epoch in range(E):

    # Predict for all samples
    y_pred = X @ theta

    # Calculate MSE (optional, for monitoring)
    loss = mean((y_pred - y)**2)

    # Calculate average gradient (By average it means we are taking the sum of all gradients and dividing by the number of samples)
    gradient = (2/m) * X.T @ (y_pred - y)

    # Update weights once
    theta = theta - alpha * gradient
```

## 2. Stochastic Gradient Descent (SGD)

- Uses **only one training sample at a time** to calculate the gradient.
- Updates weights immediately after processing each sample.
- The next sample uses the already updated weights.

### Pseudocode

```python
Initialize weights theta
alpha = 0.01

for epoch in range(E):

    Shuffle the training dataset

    for i in range(m):

        # Predict for one sample
        y_pred = X[i] @ theta

        # Calculate individual sample loss (optional)
        loss = (y_pred - y[i])**2

        # Calculate gradient for one sample
        gradient = 2 * (y_pred - y[i]) * X[i]

        # Update weights immediately
        theta = theta - alpha * gradient
```

## 3. Understanding with an Example

Suppose we have 3 training samples:

| Sample | Actual y |
|---|---:|
| 1 | 10 |
| 2 | 20 |
| 3 | 30 |

### Batch Gradient Descent

1. Predict all 3 samples using the current weights.
2. Calculate the gradient for each sample.
3. Take the average of all 3 gradients.
4. Update weights only once.

$$
G_{avg}=\frac{G_1+G_2+G_3}{3}
$$

$$
\theta_{new}=\theta_{old}-\alpha G_{avg}
$$

### Stochastic Gradient Descent

1. Take sample 1, calculate its gradient and update weights.
2. Take sample 2 using the updated weights and update again.
3. Take sample 3 using the latest weights and update again.

$$
\theta_1=\theta_0-\alpha G_1
$$

$$
\theta_2=\theta_1-\alpha G_2
$$

$$
\theta_3=\theta_2-\alpha G_3
$$

Thus, BGD performs **1 weight update**, whereas SGD performs **3 weight updates** in one epoch.

## 4. Loss vs Gradient: What is the Difference?

For MSE:

$$
J(\theta)=\frac{1}{m}\sum_{i=1}^{m}(\hat y_i-y_i)^2
$$

- **Loss:** Measures how wrong the predictions are.
- **Gradient:** Tells us how to change the weights to reduce the loss.
- We don't directly use the loss value for updating weights. We use its derivative (gradient).
- The actual loss value is useful for monitoring model performance.

## 5. Final Verdict

| Parameter | Batch GD | SGD |
|---|---|---|
| Data per update | Entire dataset | One sample |
| Gradient | Average of all sample gradients | Individual sample gradient |
| Updates per epoch | 1 | m |
| Convergence | More stable | More noisy |
| Computation per update | Higher | Lower |

**Remember:**
- BGD: Calculate gradients using all samples → Update weights once.
- SGD: Calculate gradient using one sample → Update weights immediately.
- One complete pass through the dataset = One Epoch.
- SGD's individual gradient is an estimate of the overall dataset gradient.

#### So, SGD and BGD just differ in one step: Where we update the weights. BGD updates after processing all samples (i.e. use sum of all gradients divide by the number of samples), while SGD updates after each sample.
