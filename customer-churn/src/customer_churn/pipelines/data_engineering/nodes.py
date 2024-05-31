import pandas as pd

# def preprocess_data(raw_data: pd.DataFrame) -> pd.DataFrame:
#     print(type(raw_data))
#     raw_data = raw_data.dropna()
#     categorical_columns = raw_data.select_dtypes(include=['object']).columns
#     raw_data = pd.get_dummies(raw_data, columns=categorical_columns, drop_first=True)
#     return raw_data

def preprocess_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    raw_data.drop(["customerID"], inplace = True, axis = 1)
    raw_data.TotalCharges = raw_data.TotalCharges.replace(" ",np.nan)
    raw_data.TotalCharges.fillna(0, inplace = True)
    raw_data.TotalCharges = raw_data.TotalCharges.astype(float)
    
    cols1 = ['Partner', 'Dependents', 'PaperlessBilling', 'Churn', 'PhoneService']
    for col in cols1:
        raw_data[col] = raw_data[col].apply(lambda x: 0 if x == "No" else 1)
   
    raw_data.gender = raw_data.gender.apply(lambda x: 0 if x == "Male" else 1)
    raw_data.MultipleLines = raw_data.MultipleLines.map({'No phone service': 0, 'No': 0, 'Yes': 1})
    
    cols2 = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in cols2:
        raw_data[col] = raw_data[col].map({'No internet service': 0, 'No': 0, 'Yes': 1})
    
    raw_data = pd.get_dummies(raw_data, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=True)

    return raw_data
