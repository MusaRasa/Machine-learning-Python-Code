import math

"""
Adam alforithm use form two moment 

First moment: Mt+1 = Bita1 *m1 +(1-Bita1)* Gradientw J(wt)
Second moment: Vt+1 =  Bita2 V2 +(1-Bita2) * (Gradientw J(wt)) ** 2
"""
import random
import math


# -------------------------
# Generate dataset
# -------------------------

data = []

for i in range(100):
    x = random.uniform(0, 10)

    noise = random.uniform(-1, 1)

    y = 3 * x + 5 + noise

    data.append((x, y))


# -------------------------
# Split dataset
# -------------------------

random.shuffle(data)

train_data = data[:70]
validation_data = data[70:85]
test_data = data[85:]


# -------------------------
# Model parameters
# -------------------------

w = 0.0
b = 0.0


# -------------------------
# Adam settings
# -------------------------

learning_rate = 0.01

beta1 = 0.9
beta2 = 0.999

epsilon = 1e-8


mw = 0
vw = 0

mb = 0
vb = 0

t = 0


# -------------------------
# Loss function
# -------------------------

def mse(data, w, b):

    total_error = 0

    for x, y in data:

        prediction = w * x + b

        error = prediction - y

        total_error += error ** 2

    return total_error / len(data)


# -------------------------
# Training
# -------------------------

epochs = 1000

train_losses = []
validation_losses = []


for epoch in range(epochs):

    dw = 0
    db = 0

    n = len(train_data)

    # Calculate gradients

    for x, y in train_data:

        prediction = w * x + b

        error = prediction - y

        dw += 2 * x * error
        db += 2 * error

    dw /= n
    db /= n


    # Adam time step

    t += 1


    # -------------------------
    # Adam for w
    # -------------------------

    mw = beta1 * mw + (1 - beta1) * dw

    vw = beta2 * vw + (1 - beta2) * dw ** 2


    # Bias correction

    mw_hat = mw / (1 - beta1 ** t)

    vw_hat = vw / (1 - beta2 ** t)


    # Update w

    w -= learning_rate * mw_hat / (
        math.sqrt(vw_hat) + epsilon
    )


    # -------------------------
    # Adam for b
    # -------------------------

    mb = beta1 * mb + (1 - beta1) * db

    vb = beta2 * vb + (1 - beta2) * db ** 2


    # Bias correction

    mb_hat = mb / (1 - beta1 ** t)

    vb_hat = vb / (1 - beta2 ** t)


    # Update b

    b -= learning_rate * mb_hat / (
        math.sqrt(vb_hat) + epsilon
    )


    # -------------------------
    # Calculate losses
    # -------------------------

    train_loss = mse(train_data, w, b)

    validation_loss = mse(
        validation_data,
        w,
        b
    )


    train_losses.append(train_loss)

    validation_losses.append(validation_loss)


# -------------------------
# Test
# -------------------------

test_loss = mse(test_data, w, b)


print("Final model:")
print("w =", w)
print("b =", b)

print()

print("Train Loss:",
      mse(train_data, w, b))

print("Validation Loss:",
      mse(validation_data, w, b))

print("Test Loss:",
      test_loss)




import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))

plt.plot(
    train_losses,
    label="Train Loss"
)

plt.plot(
    validation_losses,
    label="Validation Loss"
)


plt.scatter(
    epochs - 1,
    test_loss,
    label="Test Loss"
)


plt.xlabel("Epoch")

plt.ylabel("MSE Loss")

plt.title("Adam Optimizer - Training")

plt.legend()

plt.grid()

plt.show()