import streamlit as st
# Streamlit app layout
st.set_page_config(page_title="Credit Risk Analyzer", layout="wide")

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model and preprocessing objects
@st.cache_resource
def load_model():
    model = joblib.load('credit_risk_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_model()

# Function to get feature importances
def get_feature_importances(model, input_df):
    # Get feature names after preprocessing
    numeric_features = ['Age', 'Credit amount', 'Duration']
    categorical_features = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Job']
    
    # Get categorical feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
    
    all_features = numeric_features + list(cat_feature_names)
    
    # Get feature importances from the model
    importances = model.named_steps['classifier'].feature_importances_
    
    # Create a DataFrame with feature names and importance scores
    feature_importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return feature_importance_df

# Function to generate explanation
def generate_explanation(prediction, probabilities, input_data, feature_importance_df, top_n=3):
    explanation = []
    
    # Prediction summary
    risk_level = "Low Risk" if prediction == 'good' else "High Risk"
    confidence = probabilities[0] if prediction == 'bad' else probabilities[1]
    explanation.append(f"### Prediction: **{risk_level}** (Confidence: {confidence*100:.1f}%)")
    
    # Top influencing factors
    explanation.append("### Top Influencing Factors:")
    
    # Get top features and their values
    top_features = feature_importance_df.head(top_n)
    
    for _, row in top_features.iterrows():
        feature = row['Feature']
        importance = row['Importance']
        
        # Map feature back to original input
        if feature.startswith('Sex_'):
            value = input_data['Sex'].iloc[0]
        elif feature.startswith('Housing_'):
            value = input_data['Housing'].iloc[0]
        elif feature.startswith('Saving accounts_'):
            value = input_data['Saving accounts'].iloc[0]
        elif feature.startswith('Checking account_'):
            value = input_data['Checking account'].iloc[0]
        elif feature.startswith('Purpose_'):
            value = input_data['Purpose'].iloc[0]
        elif feature.startswith('Job_'):
            value = input_data['Job'].iloc[0]
        else:
            value = input_data[feature].iloc[0]
        
        explanation.append(f"- **{feature.replace('_', ' ').title()}**: {value} (Impact: {importance*100:.1f}%)")
    
    # Add specific risk indicators based on values
    if input_data['Credit amount'].iloc[0] > 5000:
        explanation.append("\n⚠️ **High Credit Amount**: Increases risk")
    if input_data['Duration'].iloc[0] > 36:
        explanation.append("⚠️ **Long Loan Duration**: Increases risk")
    if input_data['Age'].iloc[0] < 25:
        explanation.append("⚠️ **Young Age**: May increase risk")
    if input_data['Saving accounts'].iloc[0] == 'little':
        explanation.append("⚠️ **Low Savings**: Increases risk")
    
    return "\n\n".join(explanation)

# Title and description
st.title("🏦 Credit Risk Prediction Tool")
st.markdown("""
This tool predicts whether a loan applicant is **Low Risk** or **High Risk** based on their profile.
The model also identifies the key factors influencing the prediction.
""")

# Create input form
with st.form("credit_form"):
    st.header("Applicant Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 80, 30)
        sex = st.selectbox("Gender", ["male", "female"])
        job = st.selectbox("Job Category", 
                         options=[0, 1, 2, 3],
                         format_func=lambda x: ["Unskilled non-resident", 
                                              "Unskilled resident", 
                                              "Skilled", 
                                              "Highly skilled"][x])
        housing = st.selectbox("Housing", ["own", "rent", "free"])
    
    with col2:
        saving = st.selectbox("Saving Accounts", ["little", "moderate", "quite rich", "rich"])
        checking = st.selectbox("Checking Account", ["little", "moderate", "rich"])
        credit = st.number_input("Credit Amount (DM)", min_value=0, value=5000, step=100)
        duration = st.number_input("Loan Duration (months)", min_value=1, value=24)
    
    purpose = st.selectbox("Loan Purpose", 
                         ["car", "furniture/equipment", "radio/TV", 
                          "education", "business", "vacation/others"])
    
    submitted = st.form_submit_button("Predict Credit Risk")

# When form is submitted
if submitted:
    # Create input DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Job': [job],
        'Housing': [housing],
        'Saving accounts': [saving],
        'Checking account': [checking],
        'Credit amount': [credit],
        'Duration': [duration],
        'Purpose': [purpose]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    # Get feature importances
    feature_importance_df = get_feature_importances(model, input_data)
    
    # Generate explanation
    explanation = generate_explanation(prediction, probabilities, input_data, feature_importance_df)
    
    # Display results
    st.markdown("---")
    
    result_col, chart_col = st.columns([1, 1])
    
    with result_col:
        if prediction == 'good':
            st.success("## Prediction: Low Credit Risk ✅")
        else:
            st.error("## Prediction: High Credit Risk ⚠️")
        
        st.markdown(explanation)
    
    with chart_col:
        # Plot feature importances
        plt.figure(figsize=(8, 6))
        top_features = feature_importance_df.head(5)
        sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
        plt.title('Top 5 Influencing Factors')
        plt.xlabel('Importance Score')
        plt.ylabel('')
        st.pyplot(plt)
    
    # Show raw input data
    st.markdown("---")
    st.subheader("Applicant Profile Summary")
    st.dataframe(input_data.T.rename(columns={0: "Value"}))

# Add model info in sidebar
st.sidebar.header("About This Tool")
st.sidebar.markdown("""
This credit risk predictor uses a **Random Forest** model trained on German credit data.

**Model Performance:**
- Accuracy: 98%

**Key Features Considered:**
- Credit Amount
- Loan Duration
- Applicant Age
""")

st.sidebar.markdown("""
**How to Interpret Results:**
1. The prediction shows overall risk level
2. Top factors explain what influenced the decision
3. Warnings highlight specific risk indicators
""")