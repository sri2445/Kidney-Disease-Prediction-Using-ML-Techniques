import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from flask import Flask, request, render_template

app = Flask(__name__)

# Global variable to hold the trained model
xgb_model = None

def load_data(file_path):
    """Load the CSV file"""
    return pd.read_csv(file_path)

def drop_id_column(df):
    """Drop the id column"""
    df.drop('id', axis=1, inplace=True)
    return df

def rename_columns(df):
    """Rename columns to make them more user-friendly"""
    df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
                'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
                'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
                'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
                'aanemia', 'class']
    return df

def convert_to_numeric(df):
    """Convert necessary columns to numeric type"""
    df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce')
    df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce')
    df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')
    return df

def extract_categorical_and_numerical_columns(df):
    """Extract categorical and numerical columns"""
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    numerical_columns = [col for col in df.columns if df[col].dtype != 'object']
    return categorical_columns, numerical_columns

def replace_incorrect_values(df):
    """Replace incorrect values"""
    df.replace({'diabetes_mellitus': {'\tno':'no','\tyes':'yes',' yes':'yes'}}, inplace=True)
    df.replace({'coronary_artery_disease': '\tno'}, {'coronary_artery_disease': 'no'}, inplace=True)
    df['class'] = df['class'].replace(to_replace={'ckd\t': 'ckd', 'notckd': 'not ckd'})
    df['class'] = df['class'].map({'ckd': 0, 'not ckd': 1})
    df['class'] = pd.to_numeric(df['class'], errors='coerce')
    return df

def impute_missing_values(df, categorical_columns, numerical_columns):
    """Impute missing values using random sampling and mean/mode sampling"""
    def random_value_imputation(feature):
        random_sample = df[feature].dropna().sample(df[feature].isna().sum())
        random_sample.index = df[df[feature].isnull()].index
        df.loc[df[feature].isnull(), feature] = random_sample

    def impute_mode(feature):
        mode = df[feature].mode()[0]
        df[feature] = df[feature].fillna(mode)

    for col in numerical_columns:
        random_value_imputation(col)
    random_value_imputation('red_blood_cells')
    random_value_imputation('pus_cell')

    for col in categorical_columns:
        impute_mode(col)
    return df

def encode_categorical_columns(df, categorical_columns):
    """Encode categorical columns using LabelEncoder"""
    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
    return df

def load_and_process_data(file_path):
    """Load and process the data"""
    df = load_data(file_path)
    df = drop_id_column(df)
    df = rename_columns(df)
    df = convert_to_numeric(df)
    categorical_columns, numerical_columns = extract_categorical_and_numerical_columns(df)
    df = replace_incorrect_values(df)
    df = impute_missing_values(df, categorical_columns, numerical_columns)
    df = encode_categorical_columns(df, categorical_columns)
    return df

def split_data(df):
    """Split data into training and test sets"""
    ind_col = [col for col in df.columns if col != 'class']
    dep_col = 'class'
    X = df[ind_col]
    y = df[dep_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    return X_train, X_test, y_train, y_test

def build_model(X_train, y_train):
    """Build a logistic regression model"""
    # Hyperparameters:
    # - objective: binary:logistic for binary classification
    # - learning_rate: 0.5 for moderate learning rate
    # - max_depth: 5 for moderate tree depth
    # - n_estimators: 150 for moderate number of trees
    xgb = XGBClassifier(objective = 'binary:logistic', learning_rate = 0.5, max_depth = 5, n_estimators = 150)
    xgb.fit(X_train, y_train)
    return xgb

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using accuracy score, classification report, and confusion matrix"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred= None
    return accuracy

def collect(data):
    global xgb_model
    
    # Create a DataFrame from the input data
    df = pd.DataFrame([data], columns=['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',  
                                        'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria', 
                                        'blood_glucose_random', 'blood_urea', 'serum_creatinine', 
                                        'sodium', 'potassium', 'haemoglobin', 'packed_cell_volume', 
                                        'white_blood_cell_count', 'red_blood_cell_count', 'hypertension', 
                                        'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 
                                        'peda_edema', 'aanemia'])

    # Convert categorical values to 0 or 1
    categorical_columns = ['red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria', 'hypertension', 
                            'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema', 'aanemia']
    for col in categorical_columns:
        df[col] = df[col].map({'abnormal': 0, 'normal': 1, 'present': 1, 'notpresent': 0, 'yes': 1, 'no': 0, 
                                'poor': 1, 'good': 0})

    # Convert numeric values to float
    numeric_columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 
                        'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium', 'potassium', 
                        'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']
    for col in numeric_columns:
        df[col] = df[col].astype(float)

    # Make a prediction using the input DataFrame
    y_pred = xgb_model.predict(df)
    return y_pred

@app.route('/', methods=['GET', 'POST'])
def index():
    global xgb_model
    df = load_and_process_data('./datasets/kidney_disease.csv')
    X_train, X_test, y_train, y_test = split_data(df)
    xgb_model = build_model(X_train, y_train)
    accuracy = evaluate_model(xgb_model, X_test, y_test)
    return render_template('index.html', accuracy=accuracy)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

@app.route('/data', methods=['POST'])
def data():
    return render_template('data.html')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

@app.route('/prediction', methods=['POST'])
def prediction():
    if request.method == 'POST':
        # Collecting data from the form
        age = request.form.get('age')
        bp = request.form.get('bp')
        sg = request.form.get('sg')
        al = request.form.get('al')
        su = request.form.get('su')
        rbc = request.form.get('rbc')
        pc = request.form.get('pc')
        pcc = request.form.get('pcc')
        ba = request.form.get('ba')
        bgr = request.form.get('bgr')
        bu = request.form.get('bu')
        sc = request.form.get('sc')
        sod = request.form.get('sod')
        pot = request.form.get('pot')
        hemo = request.form.get('hemo')
        pcv = request.form.get('pcv')
        wc = request.form.get('wc')
        rc = request.form.get('rc')
        htn = request.form.get('htn')
        dm = request.form.get('dm')
        cad = request.form.get('cad')
        appet = request.form.get('appet')
        pe = request.form.get('pe')
        ane = request.form.get('ane')

        # Storing all values in a tuple
        data = (
            age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, 
            hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane
        )
        print(data)
        y_pred = collect(data)
        print(y_pred)
        result = y_pred  # Store the prediction result
        y_pred = None 
    return render_template('prediction.html', y_pred=result)

if __name__ == '__main__':
    app.run(debug=True)