import pandas as pd
import numpy as np
import pickle
import streamlit as st
import os
from openai import OpenAI
import utils as ut
from sklearn.ensemble import VotingClassifier

client = OpenAI(
  base_url="https://api.groq.com/openai/v1",
  api_key=os.environ.get("GROQ_API_KEY")
)



def load_model(filename):
  with open(filename, "rb") as file:
    return pickle.load(file)

xgboost_model = load_model('xgb_model.pkl')
naive_bayes_model = load_model('nb_model.pkl')
random_forest_model = load_model('rf_model.pkl')
decision_tree_model = load_model('dt_model.pkl')
svm_model = load_model('svm_model.pkl')
knn_model = load_model('knn_model.pkl')
voting_classifier_model = load_model('voting_clf.pkl')
xgboost_SMOTE_model = load_model('xgboost-SMOTE.pkl')
xgboost_featureEngineered_model = load_model('xgboost-featureEngineered.pkl')


def prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary):

  input_dict = {
    'CreditScore': credit_score,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': int(has_credit_card),
    'IsActiveMember': int(is_active_member),
    'EstimatedSalary': estimated_salary,
    'Geography_France': 1 if location == "France" else 0,
    'Geography_Germany': 1 if location == "Germany" else 0,
    'Geography_Spain': 1 if location == "Spain" else 0,
    'Gender_Male': 1 if gender == "Male" else 0,
    'Gender_Female': 1 if gender == "Female" else 0
  }

  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict

def voting_classifier_predict_proba(voting_classifier, input_df):
  estimator_probs = []

  for estimator in voting_classifier.estimators_:
      if hasattr(estimator, "predict_proba"):
          # If estimator supports predict_proba, use it
          estimator_probs.append(estimator.predict_proba(input_df)[0][1])
      elif hasattr(estimator, "decision_function"):
          # If not, use decision_function as a fallback and map to probability
          decision_score = estimator.decision_function(input_df)[0]
          prob = 1 / (1 + np.exp(-decision_score))  # Logistic function
          estimator_probs.append(prob)
      else:
          # If neither predict_proba nor decision_function is available, skip this estimator
          continue

  # Average the individual probabilities or confidence scores
  return np.mean(estimator_probs) if estimator_probs else None



def make_predictions(input_df, input_dict):

  probabilities = {
    'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
    #'Naive Bayes': naive_bayes_model.predict_proba(input_df)[0][1],
    'RandomForest': random_forest_model.predict_proba(input_df)[0][1],
    #'DecisionTree': decision_tree_model.predict_proba(input_df)[0][1],
    #'SVM': svm_model.decision_function(input_df)[0],
    #'K-Nearest Neighbors:': knn_model.predict_proba(input_df)[0][1],
    #'VotingClassifier': voting_classifier_model.predict_proba(input_df)[0][1],
    #'XGB-SMOTE': xgboost_SMOTE_model.predict(input_df)[0],
    #'XGB-features': xgboost_featureEngineered_model.predict_proba(input_df)[0][1],
  }

  avg_probability = np.mean(list(probabilities.values()))

  col1, col2 = st.columns(2)

  with col1:
    fig = ut.create_gauge_chart(avg_probability)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"The customer has a {avg_probability: .2%} probabiliity of churning.")

  with col2:
    fig_probs = ut.create_model_probability_chart(probabilities)
    st.plotly_chart(fig_probs, use_container_width=True)

  return avg_probability

  

  
def explain_prediction(probability, input_dict, surname):
  prompt = f"""You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models.
  
  Your machine learning model has predicted that a customer named {surname} has a {round(probability * 100, 1)}% probability of churning, based on the information provided below.
    
  Here is the customer's information:
  {input_dict}
      
  Here are the machine learning model's top 10 most important features for predicting churn:
      
              Feature | Importance
    --------------------------------------
       NumOfProducts  |  0.323888
      IsActiveMember  |  0.164146
                 Age  |  0.109550
   Geography_Germany  |  0.091373
             Balance  |  0.052786
    Geography_France  |  0.046463
       Gender_Female  |  0.045283
     Geography_Spain  |  0.036855
         CreditScore  |  0.035005
     EstimatedSalary  |  0.032655
           HasCrCard  |  0.031940
              Tenure  |  0.030054
         Gender_Male  |  0.000000
   
   
  {pd.set_option('display.max_columns', None)}
             
  Here are summary statistics for churned customers:
  {df[df['Exited'] == 1].describe()}
             
  Here are summary statistics for non-churned customers:
  {df[df['Exited'] == 0].describe()}
             
             
 - If the customer has over a 40% risk of churning, generate a 3 sentence explanation of why they are at risk of churning.
 - If the customer has less than a 40% risk of churning, generate a 3 sentence explanation of why they might not be at risk of churning.
 - Focus on key insights from the customer’s information, statistical comparisons with churned and non-churned customers, and feature importance.
 - Make sure to correctly use the surname and ID of the customer, just use the single name provided.
 - Do not use extraneous phrases like "here is a 3 sentence explanation of why they are at risk of churning".
 
 
 - Don't mention the probability of churning, or the machine learning model, or say anything like "Based on the machine learning model's predictions and top 10 most important features", just explain the prediction.
 
  """

  print("EXPLANATION PROMPT", prompt)
  
  raw_response = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[{
      "role": "user",
      "content": prompt
    }]
  )
  return raw_response.choices[0].message.content

def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""You are a manager at HS Bank. You are responsible for ensuring customers stay with the bank and are incentivized with various offers.
    
    You notices a customer named {surname} has a {round(probability * 100, 1)}% probability of churning.
    
    Here is the customer's information:
    {input_dict}
    
    Here is some explanation as to why the customer might be at risk of churning:
    {explanation}
    
    Generate an email to the customer based on their information, asking them to stay if they are at rish of churning, or offering them incentives so that they become more loyal to the bank. Incentives should be relevant to the customer's information, but do not mention the specifics of their information.
    
    Make sure to list out a set of incentives to stay based on their information, in bullet point format, and separate them with new lines. Don't ever mention the probability of churning, or the machine learning model, or say anything like "Based on the machine learning model's predictions".

    At the end of the email, state your title as "John Doe, Manager at HS Bank".
    """

    raw_response = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[{
      "role": "user",
      "content": prompt
      }],
    )
    print ("\n\nEMAIL PROMPT", prompt)
    return raw_response.choices[0].message.content


st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]
selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:

  selected_customer_id = int(selected_customer_option.split(" - ")[0])
  print("Selected Customer ID", selected_customer_id)

  selected_surname = selected_customer_option.split(" - ")[1]
  print("Surname", selected_surname)

  selected_customer = df.loc[df['CustomerId'] == selected_customer_id]
  print("Selected Customer", selected_customer)

  col1, col2 = st.columns(2)

  with col1:

    credit_score = st.number_input(
      "Credit Score",
      min_value=300,
      max_value=850,
      value=int(selected_customer['CreditScore']))

    location = st.selectbox(
      "Location", ["Spain", "France", "Germany"],
      index=["Spain", "France", "Germany"].index(
        selected_customer['Geography'].iloc[0]))

    gender = st.radio("Gender", ["Male", "Female"],
                     index=0 if selected_customer['Gender'].iloc[0] == "Male" else
                      1)

    age = st.number_input(
      "Age",
      min_value=18,
      max_value=100,
      value=int(selected_customer['Age'].iloc[0]))
    
    tenure = st.number_input(
      "Tenure (years)",
      min_value=0,
      max_value=50,
      value=int(selected_customer['Tenure'].iloc[0]))

  with col2:

    balance = st.number_input(
      "Balance",
      min_value=0.0,
      value=float(selected_customer['Balance'].iloc[0]))

    num_products = st.number_input(
      "Number of Products",
      min_value=1,
      max_value=10,
      value=int(selected_customer['NumOfProducts'].iloc[0]))

    has_credit_card = st.checkbox(
      "Has Credit Card",
      value=bool(selected_customer['HasCrCard'].iloc[0]))

    is_active_member = st.checkbox(
      "Is Active Member",
      value=bool(selected_customer['IsActiveMember'].iloc[0]))

    estimated_salary = st.number_input(
      "Estimated salary",
      min_value=0.0,
      value=float(selected_customer['EstimatedSalary'].iloc[0]))


  input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)

  avg_probability = make_predictions(input_df, input_dict)

  explanation = explain_prediction(avg_probability, input_dict, selected_customer['Surname'])
  st.markdown("---")
  st.subheader("Explanation of Prediction")
  st.markdown(explanation)

  email = generate_email(avg_probability, input_dict, explanation, selected_customer['Surname'])
  st.markdown("---")
  st.subheader("Personalized Email")
  st.markdown(email)