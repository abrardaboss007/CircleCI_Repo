import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import random

# Streamlit app title
st.title("Line of Best Fit Visualisation")

# Load CSV file
df = pd.read_csv("Data_Pipeline_Activity/synthetic_data.csv")

st.markdown("Hey Ed! This is the Data Pipeline Activity. As you can see below, the default slope is **3** and the default y-intercept is **7**, with the true values for these in the graph below (due to having approximated authentic noise within the y-values). Feel free to change the slope and y-intercept values to whatever you like. Random y-values will be generated accordingly. The x-values are fixed from -5 to 5 inclusive.")

# Make 2 columns for user to input desired gradient and y-intercept
col  = st.columns(2)
with col[0]:
    with st.container(border= True):
        gradient_input = st.number_input(label= "What is the gradient?", value= 3.00, format= "%0.2f")

with col[1]:
    with st.container(border= True):
       y_intercept_input = st.number_input(label= "What is the y-intercept?", value= 7.00, format="%0.2f")


# Convert initial x and y values to numpy array (where y = 3x+7)
# 3 and 7 are arbitrary numbers that the user will be met with initially 
x_values = df["x_values"].to_numpy()

# Replace all of the y-values based on the user input
def random_y_values(x):
    # Function to generate a random number between specified max and min values
    # function = rand * (max - min) + min 
    # avg = mx + c
    # max = mx + c + 1.5 * gradient_input
    # min = mx + c - 1.5 * gradient_input (ensure y_max and y_min are within 3 * gradient_input's of the true value)
    # Therefore in my case...
    # function = rand * [(mx + c + 1.5 * gradient_input) - (mx + c - 1.5 * gradient_input)] + (mx + c - 1.5 * gradient_input)
    #          = rand * [3 * gradient_input] + (mx + c - 1.5 * gradient_input)
    # where rand is a random float between 0 and 1
    y_random = (random.random() * (3 * gradient_input)) + gradient_input * x + y_intercept_input - (1.5 * gradient_input)
    return y_random

df["y_values"] = df["x_values"].apply(random_y_values)
df.to_csv("Data_Pipeline_Activity/synthetic_data.csv", index=False)
# Replace y_values array with new y_values
y_values = df["y_values"].to_numpy()


# Fit line of best fit
coefficients = np.polyfit(x_values, y_values, 1)
slope = coefficients[0]
intercept = coefficients[1]

# Get correlation coefficient
correlation_matrix = np.corrcoef(x_values,y_values)
correlation_coef = round(correlation_matrix[0][1],3)


# Create and display the plot
fig, ax = plt.subplots()
ax.scatter(x_values, y_values, marker="x")
y_fit = slope * x_values + intercept
ax.plot(x_values, y_fit, color='red', linewidth=2)

# Display true slope and intercept values due to the approximation of authentic noise in the data
ax.text(
    0.05, 0.95,  # x, y in axes coordinates (5% from left, 95% from bottom)
    f"Actual gradient = {slope:.2f}\nActual Y-intercept = {intercept:.2f}\nr = {correlation_coef}", 
    transform=ax.transAxes,  # Use axes coordinate system
    fontsize=10,
    verticalalignment='top',  # Align text top so text stays below y=0.95
    bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

ax.set_title(f"Graph of y = {round(gradient_input,2)}x + {round(y_intercept_input,2)} ")
ax.set_xlabel("X values")
ax.set_ylabel("Y values")
st.pyplot(fig)