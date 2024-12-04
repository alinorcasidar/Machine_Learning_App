import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data from CSV
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Apply log transformation
def log_transform_target(y):
    return np.log1p(y)

# Main Streamlit app
def main():
    st.title("Program Budget Predictor")
    
    # Sidebar for inputs
    st.sidebar.header("User Input")
    participants = st.sidebar.number_input("Enter the number of participants:", min_value=1, value=30)
    duration = st.sidebar.number_input("Enter the duration of the program (in hours):", min_value=1.0, value=10.0)
    staffs = st.sidebar.number_input("Enter the number of staff members:", min_value=1, value=12)
    
    program_competition = st.sidebar.checkbox("Competition Program", value=True)
    program_modeling = st.sidebar.checkbox("Modeling Program", value=False)
    program_seminar = st.sidebar.checkbox("Seminar Program", value=False)
    program_sport = st.sidebar.checkbox("Sport Program", value=False)
    program_training = st.sidebar.checkbox("Training Program", value=False)
    program_workshop = st.sidebar.checkbox("Workshop Program", value=False)
    
    month = st.sidebar.selectbox(
        "Select the month:",
        ['January', 'February', 'March', 'April', 'May', 'June', 
         'July', 'August', 'September', 'October', 'November', 'December']
    )
    
    # Load data
    data_file = r'C:\Users\casid\OneDrive\Desktop\JUPYTER NOTEBOOK\PROGRAM_TAGOLOAN_DATA.csv'
    data = load_data(data_file)
    
    # Preprocess the data
    X = data[['Number of Participants', 'Duration of Program/HR', 'Staffs',
              'Program_Competition', 'Program_Modeling', 'Program_Seminar',
              'Program_Sport', 'Program_Training', 'Program_Workshop',
              'Month_January', 'Month_February', 'Month_March',
              'Month_April', 'Month_May', 'Month_June', 'Month_July',
              'Month_August', 'Month_September', 'Month_October',
              'Month_November', 'Month_December']]
    
    y = data['Budget']
    y_transformed = log_transform_target(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=42)
    
    # Train the model
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    
    # Prepare user input
    month_one_hot = [1 if m.lower() == month.lower() else 0 for m in 
                     ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']]
    
    input_data = [[
        participants, duration, staffs,
        int(program_competition), int(program_modeling), int(program_seminar),
        int(program_sport), int(program_training), int(program_workshop),
        *month_one_hot
    ]]
    
    input_df = pd.DataFrame(input_data, columns=X.columns)
    
    # Make prediction
    prediction = model.predict(input_df)
    predicted_budget = np.expm1(prediction[0])  # Reverse log transformation
    
    # Evaluate model
    test_predictions = model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    # Display results
    st.write(f"### Predicted Budget: P{predicted_budget:.2f}")
    st.write(f"### Model Performance on Test Set:")
    st.write(f"- Mean Squared Error (MSE): {test_mse:.2f}")
    st.write(f"- R² Score: {test_r2:.2f}")

    # 1. Feature Importance Graph
    feature_importance = model.feature_importances_
    feature_names = X.columns
    fig1, ax1 = plt.subplots()
    ax1.barh(feature_names, feature_importance)
    ax1.set_title("Feature Importance")
    ax1.set_xlabel("Importance Score")
    st.pyplot(fig1)
    
    # 2. Actual vs Predicted Budget (Scatter Plot)
    actual_budgets = np.expm1(y_test)
    predicted_budgets = np.expm1(test_predictions)
    fig2, ax2 = plt.subplots()
    ax2.scatter(actual_budgets, predicted_budgets, alpha=0.5)
    ax2.plot([actual_budgets.min(), actual_budgets.max()], 
             [actual_budgets.min(), actual_budgets.max()], 
             'r--', label="Perfect Prediction")
    ax2.set_title("Actual vs Predicted Budget")
    ax2.set_xlabel("Actual Budget")
    ax2.set_ylabel("Predicted Budget")
    ax2.legend()
    st.pyplot(fig2)
    
    # 3. Error Distribution (Histogram)
    errors = actual_budgets - predicted_budgets
    fig3, ax3 = plt.subplots()
    ax3.hist(errors, bins=20, alpha=0.7, edgecolor='black')
    ax3.set_title("Distribution of Prediction Errors")
    ax3.set_xlabel("Error (Actual - Predicted)")
    ax3.set_ylabel("Frequency")
    st.pyplot(fig3)
    
    # 4. Predicted Budget Over Months (Bar Chart)
    monthly_predictions = []
    for m in ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December']:
        input_data[0][-12:] = [1 if m == month else 0 for month in 
                               ['January', 'February', 'March', 'April', 'May', 'June', 
                                'July', 'August', 'September', 'October', 'November', 'December']]
        pred = model.predict(pd.DataFrame(input_data, columns=X.columns))
        monthly_predictions.append(np.expm1(pred[0]))
    
    fig4, ax4 = plt.subplots()
    ax4.bar(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 
             'July', 'Augt', 'Sept', 'Oct', 'Nov', 'Dec'], 
            monthly_predictions)
    ax4.set_title("Predicted Budget by Month")
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Predicted Budget")
    st.pyplot(fig4)

if __name__ == "__main__":
    main()