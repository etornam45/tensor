---
title: "Machine Learning from scratch"
description: "Building neural network with linear algebra and calculus"
date: "Mar 9 2026"
---

In this blog post I will explain neural networks from scratch using linear regression. I assume you have a basic understanding of `linear algebra` and `calculus` or `multivariate calculus`. This blog looks at machine learning from a mathematical perspective. I prefer not to write this blog with any structure but with the flow of thinking.

Say you have been given a table of values that contains the `x` and `y` values of a linear function $y = mx + c$. How can we find the gradient $m$ and intercept $c$ of $y$?

| x  | y  |
:--- | -: |
| 1  | 3  |
| 2  | 5  |
| 3  | 7  |
| 4  | 9  |
| 5  | 11 |

Because we know it is a linear function we can just calculate the gradient with $m = \frac{y_2-y_1}{x_2-x_1}$. We can easily calculate the intercept by substituting $(x_1, y_1)$ and $m$ into $y = mx + c$.

**Example:** using $(1,3)$ and $(2,5)$

1. Finding the gradient, $m$

$$
m = \frac{5 - 3}{2 - 1}
$$

$$
m = 2
$$

2. Finding the intercept, $c$, using $(1, 3)$ and $m = 2$

$$
3 = 2 \times 1 + c
$$

$$
c = 1
$$

The result is $y = 2x + 1$.

Because we know that our data comes from a linear function we can easily use this algorithm to find the underlying function that produced the data. But in the real world you cannot just infer what the underlying function is by just looking at the data, so can we come up with an algorithm that helps us approximate it? Yes! That's when a combination of `backpropagation` and `gradient descent` comes to play.

We will use the two to find the underlying function of our data above. To do backpropagation you have to recollect your knowledge of partial differentiation and the chain rule.

> We are going to use the same assumption from earlier that our function is linear. Later we'll introduce neural networks to approximate more complex functions.

That said let's start with $y = mx + c$. We usually start with random numbers for $m$ and $c$, say $m = 5$ and $c = 4$, to get an initial function $y = 5x + 4$. Let's try using this function with $x \in \{1, 2\}$ to see how well it does.

| x  | new_y | y  |
:--- | ----: | -: |
| 1  | 9     | 3  |
| 2  | 14    | 5  |

We can see that our initial function performed poorly on producing the original function $y$. So we need a way to tell how bad our new function has performed on the data. We can use a `cost function` or a `loss function` to tell us how bad we did. Here we will use `Mean Squared Error` <br/> $L(y^*, y) = \frac{1}{2}(y^* - y)^2$ but note this is not the only loss function, `Mean Absolute Error` is another one.

So what is our loss given $x = 1$? Our $y^* = 9$ and $y = 3$ so our loss is:

$$
L(9, 3) = 0.5(9 - 3)^2 = 18
$$

Now that we know how bad we performed we can use backpropagation (Chain Rule) to find how (gradient -> direction and magnitude) to change our $m$ and $c$, $m = 5$ and $c = 4$. To change $m$ or $c$ `w.r.t` the loss, $L$, we need to find the chain of gradients to multiply.

$$
\frac{\partial L}{\partial y^*} = y^* - y
$$

We can also calculate the gradient of $y^*$ `w.r.t` $m$ and $c$:

$$
\frac{\partial y^*}{\partial m} = x
$$

$$
\frac{\partial y^*}{\partial c} = 1
$$

Now the `Chain Rule` for $\frac{\partial L}{\partial m}$ and $\frac{\partial L}{\partial c}$:

$$
\frac{\partial L}{\partial m} = \frac{\partial L}{\partial y^*} \times \frac{\partial y^*}{\partial m}
$$

$$
\frac{\partial L}{\partial c} = \frac{\partial L}{\partial y^*} \times \frac{\partial y^*}{\partial c}
$$

Now that we have the loss $L$ `w.r.t` $m$ and $c$. We'll move on to the last algorithm `Gradient Descent`, it is the simplest of all the algorithms. Here is the definition of gradient descent:

$$
W \leftarrow W - \eta \frac{\partial L}{\partial W}
$$

where $\eta$ is the learning rate and $W$ is the parameter we are trying to optimize, in this case $m$ and $c$. Using the values and formulas above we can calculate the new values for $m$ and $c$ with a learning rate of $\eta = 0.25$.

> The larger the learning rate the faster to get to the expected parameter but you can easily overshoot and never get to the expected parameter and vice versa.

For $m$:

$$
m = m - \eta \frac{\partial L}{\partial m}
$$

$$
m = m - \eta \left(\frac{\partial L}{\partial y^*} \times \frac{\partial y^*}{\partial m}\right)
$$

$$
m = m - \eta \left((y^* - y) \times x\right)
$$

$$
m = 5 - 0.25 \times (9 - 3) \times 1 = (5 - 1.5) = 3.5
$$

For $c$:

$$
c = c - \eta \frac{\partial L}{\partial c}
$$

$$
c = c - \eta \left(\frac{\partial L}{\partial y^*} \times \frac{\partial y^*}{\partial c}\right)
$$

$$
c = c - \eta \left((y^* - y) \times 1\right)
$$

$$
c = 4 - 0.25 \times (9 - 3) \times 1 = (4 - 1.5) = 2.5
$$

So our new function is $y = 3.5x + 2.5$. I hope you can see the pattern here (the parameters $m$ and $c$ are getting closer to the expected values $m = 2$ and $c = 1$). We can repeat the process until we get the expected values.

Before moving to step 2, let's tabulate the values of $m$ and $c$ and the loss $L$ for each iteration:

| Iteration | m     | c     | L   |
| :---      | ----: | ----: | --: |
| 1         | 5     | 4     | 18  |
| 2         | 3.5   | 2.5   |     |

The new function now gives:

| x    | new_y | y  |
| :--- | ----: | -: |
| 1    | 6     | 3  |
| 2    | 9.5   | 5  |

We will use $x = 2$ for the step 2 calculation.

> Using different values of x is done for more complex functions to get a better approximation of the underlying function.

So what is our loss for $x = 2$? Our $y^* = 9.5$ and $y = 5$ so our loss is:

$$
L(9.5, 5) = 0.5(9.5 - 5)^2 = 10.125
$$

Now we can calculate the new values for $m$ and $c$ with a learning rate of $\eta = 0.25$.

For $m$:

$$
m = m - \eta \frac{\partial L}{\partial m}
$$

$$
m = m - \eta \left(\frac{\partial L}{\partial y^*} \times \frac{\partial y^*}{\partial m}\right)
$$

$$
m = m - \eta \left((y^* - y) \times x\right)
$$

$$
m = 3.5 - 0.25 \times (9.5 - 5) \times 2 = (3.5 - 2.25) = 1.25
$$

For $c$:

$$
c = c - \eta \frac{\partial L}{\partial c}
$$

$$
c = c - \eta \left(\frac{\partial L}{\partial y^*} \times \frac{\partial y^*}{\partial c}\right)
$$

$$
c = c - \eta \left((y^* - y) \times 1\right)
$$

$$
c = 2.5 - 0.25 \times (9.5 - 5) \times 1 = (2.5 - 1.125) = 1.375
$$

So our new function is $y = 1.25x + 1.375$.

Step 3

| Iteration | m     | c     | L      |
| :---      | ----: | ----: | -----: |
| 1         | 5     | 4     | 18     |
| 2         | 3.5   | 2.5   | 10.125 |
| 3         | 1.25  | 1.375 |        |

Our new function now gives:

| x    | new_y  | y  |
| :--- | -----: | -: |
| 1    | 2.625  | 3  |
| 2    | 3.875  | 5  |
| 3    | 5.125  | 7  |

Repeating the process over and over again will converge to the approximate values of $m = 2$ and $c = 1$. We can write a computer algorithm to make the process easier rather than using a loop. I will be using Python here to make it easier to understand but you can use any language of your choice to implement it.

I will first create the data using `numpy`:

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

Now that we have the data we can proceed to randomly initializing our $m$ and $c$:

```py
import random
# 1. Setup
w = random.random()
c = random.random()

print(w)
print(c)
```
```bash
# Output example
0.123456789
0.987654321
```

Now we can proceed to the next step, which is to calculate the loss $L$ and the gradients $\frac{\partial L}{\partial m}$ and $\frac{\partial L}{\partial c}$:

```py
# 2. Calculate Loss and Gradients
# (x and y here refer to a single data point in the training loop)

# Calculate the predicted value
y_pred = w * x + c

# Calculate the loss
loss = 0.5 * (y_pred - y) ** 2

# Calculate the gradients
dloss_dy_pred = y_pred - y
dy_pred_dw = x
dy_pred_dc = 1

dloss_dw = dloss_dy_pred * dy_pred_dw
dloss_dc = dloss_dy_pred * dy_pred_dc

print(f"Loss: {loss}")
print(f"Gradient w.r.t. w: {dloss_dw}")
print(f"Gradient w.r.t. c: {dloss_dc}")
```
```bash
# Output example
Loss: 10.125
Gradient w.r.t. w: 2.25
Gradient w.r.t. c: 1.125
```

Now we can proceed to the next step, which is to update the values of $m$ and $c$ using the gradients we calculated above:

```py
# 3. Update Parameters
learning_rate = 0.25

w = w - learning_rate * dloss_dw
c = c - learning_rate * dloss_dc

print(f"Updated w: {w}")
print(f"Updated c: {c}")
```
```bash
# Output example
Updated w: 1.25
Updated c: 1.375
```

#### Putting it all together

We can now put it all together in a loop to repeat the process until we get the expected values of $m = 2$ and $c = 1$:

```py
import numpy as np
import random

X = np.arange(0, 10, 1)
Y = 2 * X + 1

# 1. Setup
W = random.random() * 10
C = random.random() * 10

print(W)
print(C)

epoch = 50  # the number of times we iterate through the dataset
learning_rate = 0.05

# 2. Loop
for i in range(epoch):
    # now we'll move through the data points one by one
    LOSS = 0
    for x, y in zip(X, Y):

        y_pred = W * x + C

        loss = 0.5 * (y_pred - y) ** 2

        LOSS += loss  # accumulate the loss

        # Calculate the gradients
        dloss_dy_pred = y_pred - y
        dy_pred_dw = x
        dy_pred_dc = 1

        dloss_dw = dloss_dy_pred * dy_pred_dw
        dloss_dc = dloss_dy_pred * dy_pred_dc

        # Update Parameters
        W = W - learning_rate * dloss_dw
        C = C - learning_rate * dloss_dc

    print(f"Epoch = {i} Loss = {(LOSS / len(X)):.{5}f} Y = {W:.{2}f}X + {C:.{2}f}")
```

```bash
6.436103983994634
8.562471646963367
Epoch = 0 Loss = 31.27642 Y = 0.79X + 5.87
Epoch = 1 Loss = 2.18066 Y = 1.67X + 5.60
Epoch = 2 Loss = 2.53907 Y = 1.60X + 5.06
....
....
....
Epoch = 47 Loss = 0.00007 Y = 2.00X + 1.02
Epoch = 48 Loss = 0.00005 Y = 2.00X + 1.02
Epoch = 49 Loss = 0.00004 Y = 2.00X + 1.02
```

If you haven't realized it yet this is machine learning and this simple algorithm when expanded creates very intelligent machines that can recognize objects, create art, fold proteins, drive cars and the list goes on. So we can say that this algorithm is a universal function approximator.

## Summary

In this post, we built a machine learning algorithm from scratch using only linear regression as our model. Here's the key idea broken down into three steps:

1. **Forward Pass:** Given the current parameters ($m$ and $c$), compute a prediction $y^*$ and measure how wrong it is using a loss function (we used Mean Squared Error: $L = \frac{1}{2}(y^* - y)^2$).

2. **Backpropagation:** Use the Chain Rule of calculus to compute how much each parameter contributed to the error. This gives us the gradients $\frac{\partial L}{\partial m}$ and $\frac{\partial L}{\partial c}$.

3. **Gradient Descent:** Nudge each parameter in the direction that reduces the loss: $W \leftarrow W - \eta \frac{\partial L}{\partial W}$. The learning rate $\eta$ controls the size of each step.

Repeat these three steps across the dataset for many iterations (epochs), and the parameters will gradually converge to the values that best describe the underlying function.

This is the exact same loop (forward pass, backpropagation, gradient descent) that powers modern deep learning. In a neural network, the model is more complex (many layers of neurons with non-linear activation functions), but the training algorithm is fundamentally identical to what we built here. Once you understand this loop, you understand the engine behind all of modern AI.


Thank you for reading. I will be writing a part two on `Neural Networks` and `Computation Graphs`