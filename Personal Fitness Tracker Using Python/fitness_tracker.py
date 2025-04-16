import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("🏃‍♂️ Personal Fitness Tracker")
st.markdown("""
### Predict Calories Burned Based on Your Workout
In this WebApp, you will be able to observe your predicted calories burned in your body.
Pass your parameters such as Age, Gender, BMI, etc., into this WebApp and then you will see the predicted value in kilocalories burned.
""")

# Load Data Function
@st.cache_data
def load_data():
    calories_df = pd.read_csv("calories.csv")
    exercise_df = pd.read_csv("exercise.csv")
    
    # Merge both datasets on User_ID
    df = pd.merge(exercise_df, calories_df, on="User_ID")
    
    # Gender ko numeric bana do
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    return df

# Load the dataset
df = load_data()

# Sidebar for user input
st.sidebar.subheader("🎯 Select Your Input")

age = st.sidebar.slider("Age", 18, 60, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
height = st.sidebar.slider("Height (cm)", 150, 200, 170)
weight = st.sidebar.slider("Weight (kg)", 40, 120, 65)
duration = st.sidebar.slider("Duration (mins)", 5, 120, 30)
heart_rate = st.sidebar.slider("Heart Rate", 60, 180, 100)
body_temp = st.sidebar.slider("Body Temperature (°C)", 36.0, 40.0, 37.0)


# Display user input
st.markdown("### Your Parameters:")
st.write(f"Age: {age}")
st.write(f"Gender: {gender}")
st.write(f"Height: {height} cm")
st.write(f"Weight: {weight} kg")
st.write(f"Duration: {duration} mins")
st.write(f"Heart Rate: {heart_rate} bpm")
st.write(f"Body Temperature: {body_temp} °C")

# Data preparation
X = df.drop(['Calories', 'User_ID'], axis=1)  # Drop User_ID as it's not needed for model
y = df['Calories']

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Prepare user input for prediction
user_input = pd.DataFrame([{
    'Age': age,
    'Gender': 1 if gender == 'Male' else 0,
    'Height': height,
    'Weight': weight,
    'Duration': duration,
    'Heart_Rate': heart_rate,
    'Body_Temp': body_temp
}])

# Ensure the input matches the training data columns
user_input = user_input[['Age', 'Gender', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']]

# Predict
if st.button("💪 Predict Calories Burned"):
    prediction = model.predict(user_input)
    st.success(f"🔥 Estimated Calories Burned: {int(prediction[0])} kcal")

# Similar Results: Show comparison with similar entries from the dataset
st.markdown("### Similar Results:")
similar_results = df[(df['Age'] >= age - 5) & (df['Age'] <= age + 5) &
                     (df['Duration'] >= duration - 5) & (df['Duration'] <= duration + 5)]
st.dataframe(similar_results[['Age', 'Gender', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']].head(10))

# General Information based on user input
st.markdown("### General Information:")

age_comparison = (df['Age'] < age).mean() * 100
duration_comparison = (df['Duration'] > duration).mean() * 100
heart_rate_comparison = (df['Heart_Rate'] > heart_rate).mean() * 100
body_temp_comparison = (df['Body_Temp'] > body_temp).mean() * 100

st.write(f"You are older than {age_comparison:.2f}% of other people.")
st.write(f"Your exercise duration is higher than {duration_comparison:.2f}% of other people.")
st.write(f"You have a higher heart rate than {heart_rate_comparison:.2f}% of other people during exercise.")
st.write(f"You have a higher body temperature than {body_temp_comparison:.2f}% of other people during exercise.")

# Show dataset if user wants
if st.checkbox("📊 Show sample dataset"):
    st.dataframe(df.head(10))

# Visualization: Calories vs Duration
st.markdown("### 📈 Calories vs Duration")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='Duration', y='Calories', hue='Gender', ax=ax)
st.pyplot(fig)