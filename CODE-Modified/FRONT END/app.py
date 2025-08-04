from flask import Flask,render_template,redirect,request,url_for, send_file
import mysql.connector, re
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


app = Flask(__name__)

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3307",
    database='dia'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (email, password) VALUES (%s, %s)"
                values = (email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return render_template('home.html')
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/algorithm', methods=['GET', 'POST'])
def algorithm():
    if request.method == "POST":
        algorithm = request.form['algorithm']

        MAE = {
            'K Neighbors Regressor': 768.5879295819226,
            'Gradient Boosting Regressor': 889.0564264317449,
            'Linear Regression': 1104.7265039962642,
            'Decision Tree Regressor': 114.53964995491792,
            'Random Forest Regressor': 127.98230709997871,
            'XG BRegressor': 166.38966692866515
        }

        accuracy = MAE[algorithm]
        
        return render_template('algorithm.html', algorithm = algorithm, accuracy = accuracy)
    return render_template('algorithm.html')




# Load the dataset
df = pd.read_csv(r"Dataset\crop_yield_data.csv")

# Replace spaces in column names with underscores
df.columns = [re.sub(r'\s+', '_', col) for col in df.columns]

# Define object columns to be encoded
object_columns = df.select_dtypes(include=['object']).columns

# Store label counts before encoding
labels = {col: df[col].value_counts().to_dict() for col in object_columns}

# Initialize LabelEncoder
le = LabelEncoder()

# # Encode categorical columns and store the encoded value counts
encodes = {}
for col in object_columns:
    df[col] = le.fit_transform(df[col])
    value_counts = df[col].value_counts().to_dict()
    encodes[col] = value_counts

dic = {}

for key in labels.keys():
    dic[key] = []
    for sub_key, value in labels[key].items():
        for id_key, id_value in encodes[key].items():
            if value == id_value:
                dic[key].append((sub_key, id_key))
                break

# Splitting the data
X = df.drop(columns=['Yield'])
y = df['Yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 4. Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    prediction = Soil_Moisture = Pest_Infestation_Severity = recommendation_1 = recommendation_2 = estimated_income = None

    if request.method == 'POST':

        Region = int(request.form['Region'])
        Crop_Type = int(request.form['Crop_Type'])
        Temperature = float(request.form['Temperature'])
        Rainfall = float(request.form['Rainfall'])
        Pesticide_Use = float(request.form['Pesticide_Use'])
        Soil_Moisture = float(request.form['Soil_Moisture'])
        Humidity = float(request.form['Humidity'])
        Solar_Radiation = float(request.form['Solar_Radiation'])
        Wind_Speed = float(request.form['Wind_Speed'])
        Previous_Crop = int(request.form['Previous_Crop'])
        Fertilizer_Residuals = float(request.form['Fertilizer_Residuals'])
        Pest_Infestation_Severity = float(request.form['Pest_Infestation_Severity'])
        Disease_Presence = float(request.form['Disease_Presence'])
        Pesticide_Effectiveness = float(request.form['Pesticide_Effectiveness'])
        Monsoon_Delay = float(request.form['Monsoon_Delay'])
        Flood_Occurrence = float(request.form['Flood_Occurrence'])
        Temperature_Extremes = float(request.form['Temperature_Extremes'])
        Year = float(request.form['Year'])

        # Prepare input data for prediction
        input_data = np.array([[Region, Crop_Type, Temperature, Rainfall, Pesticide_Use, Soil_Moisture, Humidity, Solar_Radiation, Wind_Speed, Previous_Crop, Fertilizer_Residuals, Pest_Infestation_Severity, Disease_Presence, Pesticide_Effectiveness, Monsoon_Delay, Flood_Occurrence, Temperature_Extremes, Year]])

        predicted_yield = model.predict(input_data)
        predicted_yield_value = predicted_yield[0]  # Extracting the scalar value
        
        # Display the prediction
        prediction = f"Predicted Yield: {predicted_yield_value:.2f} kg/ha"

        # Simple recommendation logic based on current factors
        if Soil_Moisture < 20:
            recommendation_1 = "Increase irrigation to improve soil moisture levels."
        else:
            recommendation_1 = "Soil moisture levels are adequate."

        if Pest_Infestation_Severity > 5:
            recommendation_2 = "Apply additional pesticide to control pest severity."
        else:
            recommendation_2 = "Pest infestation levels are under control, no additional pesticides needed."

        # Economic estimation
        price_per_kg = 0.80  # according to current market price $0.80 per kg
        estimated_income = predicted_yield_value * price_per_kg  # Use the scalar value
        estimated_income = f"Estimated Income: ${estimated_income:.2f} (at ${price_per_kg}/kg)"

    return render_template('prediction.html', 
                           data=dic, 
                           prediction=prediction, 
                           Soil_Moisture = Soil_Moisture,
                           recommendation_1 = recommendation_1, 
                           Pest_Infestation_Severity = Pest_Infestation_Severity,
                           recommendation_2 = recommendation_2, 
                           estimated_income = estimated_income
                           )



@app.route('/graph')
def graph():
    return render_template('graph.html')


if __name__ == '__main__':
    app.run(debug = True)

