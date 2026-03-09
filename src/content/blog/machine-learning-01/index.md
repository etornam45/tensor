---
title: "Machine Learning from Scratch"
description: "Building a neural network with linear algebra and calculus"
date: "Mar 9 2026"
---

In this blog post I will explain neural networks from scratch using linear regression. I assume you have a basic understanding of `linear algebra` and `calculus` (or `multivariate calculus`). This blog looks at machine learning from a mathematical perspective. I prefer not to write this blog with any rigid structure, but rather let it follow the natural flow of thinking.

Say you have been given a table of values that contains the `x` and `y` values of a linear function $y = mx + c$. How can we find the gradient $m$ and intercept $c$ of $y$?

| x  | y  |
:--- | -: |
| 1  | 3  |
| 2  | 5  |
| 3  | 7  |
| 4  | 9  |
| 5  | 11 |

Because we know it is a linear function, we can calculate the gradient with $m = \frac{y_2-y_1}{x_2-x_1}$ and then easily find the intercept by substituting $(x_1, y_1)$ and $m$ into $y = mx + c$.

**Example:** using $(1,3)$ and $(2,5)$

1. Finding the gradient, $m$

$$
m = \frac{5 - 3}{2 - 1} = 2
$$

2. Finding the intercept, $c$, using $(1, 3)$ and $m = 2$

$$
3 = 2 \times 1 + c \implies c = 1
$$

The result is $y = 2x + 1$.

Because we know that our data comes from a linear function, we can easily use this algorithm to find the underlying function that produced the data. But in the real world you cannot always infer what the underlying function is just by looking at the data, so can we come up with an algorithm that helps us approximate it? Yes! That's where a combination of `backpropagation` and `gradient descent` comes into play.

We will use the two to find the underlying function of our data above. To do backpropagation, you need to recall your knowledge of **partial differentiation** and the **chain rule**.

> We are going to use the same assumption from earlier, that our function is linear. Later we'll introduce neural networks to approximate more complex functions.

That said, let's start with $y = mx + c$. We usually start with random numbers for $m$ and $c$ (say $m = 5$ and $c = 4$), giving us the initial function $y = 5x + 4$. Let's try this function with $x \in \{1, 2\}$ to see how well it does.

| x  | new_y | y  |
:--- | ----: | -: |
| 1  | 9     | 3  |
| 2  | 14    | 5  |

We can see that our initial function performed poorly. So we need a way to measure how bad it did. We use a `loss function` (also called a `cost function`) for this. Here we will use **Mean Squared Error**:

$$L(y^*, y) = \frac{1}{2}(y^* - y)^2$$

Note this is not the only loss function. **Mean Absolute Error** is another common choice.

So what is our loss given $x = 1$? With $y^* = 9$ and $y = 3$:

$$
L(9, 3) = 0.5(9 - 3)^2 = 18
$$

Now that we know how poorly we performed, we can use backpropagation (the Chain Rule) to find the **gradient** (i.e., the direction and magnitude) by which to adjust $m$ and $c$.

$$
\frac{\partial L}{\partial y^*} = y^* - y
$$

We can also calculate the gradient of $y^*$ with respect to $m$ and $c$:

$$
\frac{\partial y^*}{\partial m} = x \qquad \frac{\partial y^*}{\partial c} = 1
$$

Applying the **Chain Rule**:

$$
\frac{\partial L}{\partial m} = \frac{\partial L}{\partial y^*} \times \frac{\partial y^*}{\partial m}
\qquad
\frac{\partial L}{\partial c} = \frac{\partial L}{\partial y^*} \times \frac{\partial y^*}{\partial c}
$$

Now that we have the loss with respect to $m$ and $c$, we move on to the last algorithm: **Gradient Descent**. Here is its definition:

$$
W \leftarrow W - \eta \frac{\partial L}{\partial W}
$$

where $\eta$ is the **learning rate** and $W$ is the parameter we are trying to optimize, in this case $m$ and $c$.

> The larger the learning rate, the faster we move toward the expected parameter, but if it's too large, we can overshoot and never converge. Too small and training becomes very slow. It's a balance.

Using $\eta = 0.25$ and $x = 1$:

**For $m$:**

$$
m = m - \eta \left(\frac{\partial L}{\partial y^*} \times \frac{\partial y^*}{\partial m}\right)
= m - \eta \left((y^* - y) \times x\right)
$$

$$
m = 5 - 0.25 \times (9 - 3) \times 1 = 5 - 1.5 = 3.5
$$

**For $c$:**

$$
c = c - \eta \left(\frac{\partial L}{\partial y^*} \times \frac{\partial y^*}{\partial c}\right)
= c - \eta \left((y^* - y) \times 1\right)
$$

$$
c = 4 - 0.25 \times (9 - 3) \times 1 = 4 - 1.5 = 2.5
$$

Our new function is $y = 3.5x + 2.5$. Notice that $m$ and $c$ are already getting closer to the expected values of $m = 2$ and $c = 1$.

Let's tabulate the iterations:

| Iteration | m     | c     | L      |
| :---      | ----: | ----: | -----: |
| 1         | 5     | 4     | 18     |
| 2         | 3.5   | 2.5   |        |

The new function gives:

| x    | new_y | y  |
| :--- | ----: | -: |
| 1    | 4.5   | 3  |
| 2    | 9.5   | 5  |

We'll use $x = 2$ for the next iteration.

> Using different values of $x$ across iterations is important for more complex functions, as it exposes the model to different parts of the data and leads to a better approximation.

So what is our loss for $x = 2$? With $y^* = 9.5$ and $y = 5$:

$$
L(9.5, 5) = 0.5(9.5 - 5)^2 = 10.125
$$

**For $m$:**

$$
m = 3.5 - 0.25 \times (9.5 - 5) \times 2 = 3.5 - 2.25 = 1.25
$$

**For $c$:**

$$
c = 2.5 - 0.25 \times (9.5 - 5) \times 1 = 2.5 - 1.125 = 1.375
$$

Our new function is $y = 1.25x + 1.375$.

| Iteration | m     | c     | L      |
| :---      | ----: | ----: | -----: |
| 1         | 5     | 4     | 18     |
| 2         | 3.5   | 2.5   | 10.125 |
| 3         | 1.25  | 1.375 |        |

The new function now gives:

| x    | new_y  | y  |
| :--- | -----: | -: |
| 1    | 2.625  | 3  |
| 2    | 3.875  | 5  |
| 3    | 5.125  | 7  |

Repeating this process over and over again will converge to the values $m = 2$ and $c = 1$. Rather than doing this by hand, we can write a computer algorithm using a loop. I'll be using Python here, but you can use any language of your choice.

#### Putting it all together

First, let's generate the data using `numpy`:

```py
import numpy as np

x = np.arange(0, 10, 1)
y = 2 * x + 1

print(x)
print(y)
```
```bash
# Output
[0 1 2 3 4 5 6 7 8 9]
[ 1  3  5  7  9 11 13 15 17 19]
```

Next, randomly initialize $m$ and $c$:

```py
import random

W = random.random()
C = random.random()

print(W)
print(C)
```
```bash
# Output example
0.123456789
0.987654321
```

Now we can implement the full training loop, calculating the loss, computing the gradients via backpropagation, and updating the parameters via gradient descent:

```py
import numpy as np
import random

X = np.arange(0, 10, 1)
Y = 2 * X + 1

# 1. Initialize parameters randomly
W = random.random() * 10
C = random.random() * 10

print(f"Initial W: {W:.4f}, Initial C: {C:.4f}")

epochs = 50        # number of passes through the dataset
learning_rate = 0.05

# 2. Training loop
for i in range(epochs):
    LOSS = 0
    for x, y in zip(X, Y):

        # Forward pass
        y_pred = W * x + C

        # Loss (Mean Squared Error)
        loss = 0.5 * (y_pred - y) ** 2
        LOSS += loss

        # Backpropagation (Chain Rule)
        dloss_dy_pred = y_pred - y
        dy_pred_dw = x
        dy_pred_dc = 1

        dloss_dw = dloss_dy_pred * dy_pred_dw
        dloss_dc = dloss_dy_pred * dy_pred_dc

        # Gradient Descent update
        W = W - learning_rate * dloss_dw
        C = C - learning_rate * dloss_dc

    print(f"Epoch = {i:2d} | Loss = {(LOSS / len(X)):.5f} | Y = {W:.2f}X + {C:.2f}")
```

```bash
Initial W: 6.4361, Initial C: 8.5625
Epoch =  0 | Loss = 31.27642 | Y = 0.79X + 5.87
Epoch =  1 | Loss =  2.18066 | Y = 1.67X + 5.60
Epoch =  2 | Loss =  2.53907 | Y = 1.60X + 5.06
....
....
....
Epoch = 47 | Loss =  0.00007 | Y = 2.00X + 1.02
Epoch = 48 | Loss =  0.00005 | Y = 2.00X + 1.02
Epoch = 49 | Loss =  0.00004 | Y = 2.00X + 1.01
```

If you haven't realized it yet, **this is machine learning**. This simple algorithm, when scaled up and extended, creates the intelligent systems that can recognize objects in images, generate art, fold proteins, drive cars, and much more. At its core, machine learning is about finding a good approximation to an unknown function, and gradient descent is how we get there.

## Summary

In this post, we built a machine learning algorithm from scratch using only linear regression as our model. Here's the key idea broken down into three steps:

1. **Forward Pass:** Given the current parameters ($m$ and $c$), compute a prediction $y^*$ and measure how wrong it is using a loss function (we used Mean Squared Error: $L = \frac{1}{2}(y^* - y)^2$).

2. **Backpropagation:** Use the Chain Rule of calculus to compute how much each parameter contributed to the error. This gives us the gradients $\frac{\partial L}{\partial m}$ and $\frac{\partial L}{\partial c}$.

3. **Gradient Descent:** Nudge each parameter in the direction that reduces the loss: $W \leftarrow W - \eta \frac{\partial L}{\partial W}$. The learning rate $\eta$ controls the size of each step.

Repeat these three steps across the dataset for many iterations (epochs), and the parameters will gradually converge to the values that best describe the underlying function.

This is the exact same loop (forward pass, backpropagation, gradient descent) that powers modern deep learning. In a neural network, the model is more complex (many layers of neurons with non-linear activation functions), but the training algorithm is fundamentally identical to what we built here. Once you understand this loop, you understand the engine behind all of modern AI.


I will be writing a part two soon where we'll look at neural networks and computation graphs

Thank you for reading