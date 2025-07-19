#pip install streamlit pandas numpy scikit-learn xgboost
#cd to the directory where this file is saved
#streamlit run smartbus.py
#to simply explain how the xgboost model works without jargon, it uses decision trees to predict bus demand based on various features like time of day, weather, and local events. It learns from historical data to make accurate predictions in real-time.
#decision trees are a type of model that splits data into branches based on feature values, making it easy to understand how decisions are made.
#to think of it, imagine a flowchart where each question about the data leads you down a path to a final decision, like predicting whether bus demand will be high, medium, or low based on the conditions at that time.
#the advantage of using xgboost is that it combines the predictions of many decision trees to improve accuracy and robustness, making it a powerful tool for this type of prediction task.
#xgboost is the best model for this task because it efficiently handles large datasets, captures complex patterns, and provides high accuracy in classification tasks like predicting bus demand. Its ability to combine multiple decision trees helps it learn from various features effectively, making it suitable for real-time predictions in dynamic environments like public transit.
#the reason why other models like logistic regression or random forests are not used is that logistic regression may oversimplify the relationships in the data, while random forests, although powerful, can be slower and less efficient than xgboost for large datasets. XGBoost's gradient boosting framework allows it to focus on the most important features and improve predictions iteratively, making it more effective for this specific task.
#if you want to intuitevly understand how the xgboost model works, think of it as a smart decision-making process that learns from past bus demand patterns. It uses a series of questions about the current conditions (like time of day, weather, and local events) to predict whether bus demand will be high, medium, or low. Each question narrows down the possibilities, leading to a final prediction based on the most relevant factors.
#this approach allows the model to adapt to changing conditions and make accurate predictions in real-time,

"""
Do a Live Demo: Have the app open. Say, "Let's test a scenario. Imagine it's 5 PM on a Friday in July, it's 26 degrees, and there's a concert downtown..." Change the sliders and dropdowns in real-time and click "Predict." The result will appear instantly, creating a "wow" moment.
Explain the "How": Point to the "How the AI Decides" chart. "This isn't a black box. Our model is transparent. As you can see, it has learned that the time of day and special events are the most critical factors, which aligns with common sense but also allows it to capture complex interactions."
Showcase Your Resourcefulness: Mention how you built this. "We designed the logic and architecture, and then used Streamlit—a modern tool for rapid prototyping—to build this user-friendly interface in just a few hours. This demonstrates our ability to quickly turn a complex idea into a tangible product."
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

# ==============================================================================
# App Configuration
# ==============================================================================
st.set_page_config(
    page_title="SmartBus AI for Halifax",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Model Training (Cached for Performance)
# ==============================================================================
# This function handles data creation and model training.
# @st.cache_resource ensures this complex part runs only ONCE.
@st.cache_resource
def load_model_and_data():
    """
    Generates synthetic data and trains an XGBoost model.
    Returns the trained model, label encoder, and feature columns.
    """
    # 1. Create Synthetic Data
    def create_synthetic_data(n_samples=10000):
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=n_samples, freq='h'))
        data = {
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'temperature_c': np.random.randint(-10, 28, size=n_samples),
            'precipitation_mm': np.random.choice([0, 0, 1, 5, 10], size=n_samples, p=[0.7, 0.15, 0.08, 0.05, 0.02]),
            'is_event_nearby': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
            'land_use': np.random.choice(['Downtown', 'Residential', 'University', 'Waterfront Park'], size=n_samples)
        }
        df = pd.DataFrame(data)
        conditions = [
            ((df['hour'].isin([7, 8, 16, 17, 18])) & (df['day_of_week'] < 5)),
            (df['is_event_nearby'] == 1),
            ((df['land_use'] == 'University') & (df['hour'].isin(range(9, 16))) & (df['day_of_week'] < 5)),
            ((df['land_use'] == 'Waterfront Park') & (df['temperature_c'] > 20) & (df['precipitation_mm'] == 0)),
            (df['hour'].isin(range(0, 5))),
            (df['precipitation_mm'] > 5)
        ]
        choices = ['High', 'High', 'High', 'High', 'Low', 'Low']
        df['demand_level'] = np.select(conditions, choices, default='Medium')
        return df

    df = create_synthetic_data()

    # 2. Preprocess Data
    df_processed = pd.get_dummies(df, columns=['land_use', 'day_of_week', 'month'], drop_first=False)
    X = df_processed.drop('demand_level', axis=1)
    y = df_processed['demand_level']

    # 3. Encode Target Variable and Train Model
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, _, y_train, _ = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    return model, le, X.columns

# Load the trained model, encoder, and columns
# A spinner shows a loading message while this runs for the first time
with st.spinner('Warming up the AI model... Please wait.'):
    model, le, model_columns = load_model_and_data()

# ==============================================================================
# User Interface (UI)
# ==============================================================================
st.title("🚌 SmartBus AI for Halifax")
st.markdown("A proof-of-concept to predict bus demand using real-time conditions. Set the parameters on the left and see the AI's prediction!")

# --- Sidebar for User Inputs ---
st.sidebar.header("Set Real-Time Conditions")

hour = st.sidebar.slider("Hour of the Day", 0, 23, 17)
day_name = st.sidebar.select_slider(
    "Day of the Week",
    options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    value='Friday'
)
month_name = st.sidebar.select_slider(
    "Month",
    options=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    value='Jul'
)
temperature = st.sidebar.slider("Temperature (°C)", -20, 35, 26)
precipitation = st.sidebar.slider("Precipitation (mm/hr)", 0, 20, 0)
event_status = st.sidebar.selectbox("Is there a special event nearby?", ('No', 'Yes'))
land_use_type = st.sidebar.selectbox(
    "Stop Location Type",
    ('Downtown', 'Residential', 'University', 'Waterfront Park')
)

# --- Convert UI inputs to model-ready format ---
day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

user_input = {
    'hour': hour,
    'temperature_c': temperature,
    'precipitation_mm': precipitation,
    'is_event_nearby': 1 if event_status == 'Yes' else 0,
    f'land_use_{land_use_type}': 1,
    f'day_of_week_{day_map[day_name]}': 1,
    f'month_{month_map[month_name]}': 1
}

input_df = pd.DataFrame([user_input])
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# --- Prediction Logic and Display ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Prediction")
    predict_button = st.button("Predict Bus Demand", type="primary", use_container_width=True)
    if predict_button:
        with st.spinner('AI is thinking...'):
            prediction_encoded = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            confidence = np.max(prediction_proba) * 100
            predicted_demand = le.inverse_transform(prediction_encoded)[0]

            if predicted_demand == 'High':
                st.success(f"📈 Predicted Demand: **High**")
                st.metric(label="Confidence", value=f"{confidence:.1f}%")
            elif predicted_demand == 'Medium':
                st.warning(f"📊 Predicted Demand: **Medium**")
                st.metric(label="Confidence", value=f"{confidence:.1f}%")
            else:
                st.error(f"📉 Predicted Demand: **Low**")
                st.metric(label="Confidence", value=f"{confidence:.1f}%")
            
            st.caption("This prediction helps transit dispatchers make smarter, real-time decisions.")

with col2:
    st.subheader("How the AI Decides")
    st.write("The AI weighs different factors to make its prediction. Here are the most important features it has learned:")
    
    # Feature Importance Plot
    feature_importances = pd.DataFrame({
        'feature': model_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    st.bar_chart(feature_importances.set_index('feature'))
    st.caption("For example, the model has learned that the 'hour' of the day and whether an 'event' is nearby are very strong indicators of demand.")

# --- Add a final section about the project ---
st.markdown("---")
st.markdown("This interactive app is a prototype for the SHAD Canada 2025 Design-Entrepreneurship Project.")