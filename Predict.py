import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import messagebox

# Load the dataset
data = pd.read_csv("diabetes.csv")

# Prepare the data, replacing the pedigree function with family history score for simplicity
data = data.drop('DiabetesPedigreeFunction', axis=1)

# Split the features (X) and the target (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (excluding the family history score)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Help messages for each input field
help_messages = {
    "Pregnancies": "Number of times the patient has been pregnant.",
    "Glucose": "Plasma glucose concentration (mg/dL). Higher values may indicate diabetes.",
    "BloodPressure": "Diastolic blood pressure (mm Hg). High levels may be a risk factor.",
    "SkinThickness": "Thickness of the triceps skinfold (mm). Used to estimate body fat.",
    "Insulin": "Insulin level (mu U/mL). Lower or higher levels could indicate diabetes.",
    "BMI": "Body Mass Index (weight in kg / height in m^2). Measures body fat.",
    "Family History": "Enter 0 (no family history), 1 (one relative with diabetes), or 2 (two or more relatives with diabetes).",
    "Age": "Age of the individual (years). Older individuals are at higher risk."
}

def show_help(field):
    """Show a help message for the corresponding field."""
    messagebox.showinfo(f"{field} Information", help_messages[field])

def predict_diabetes():
    """Predict if the user is diabetic based on input data."""
    try:
        # Gather inputs
        values = [
            float(entry_pregnancies.get()),
            float(entry_glucose.get()),
            float(entry_bloodpressure.get()),
            float(entry_skinthickness.get()),
            float(entry_insulin.get()),
            float(entry_bmi.get()),
            int(entry_family_history.get()),  # Family history score
            float(entry_age.get())
        ]

        # Scale the inputs (excluding family history)
        input_data = scaler.transform([values[:-2] + [values[-1]]])

        # Predict the probability and classify it
        probability = model.predict_proba(input_data)[0][1] * 100
        likelihood = "not likely" if probability < 50 else "likely"
        result = f"The probability of being diabetic is {probability:.2f}% ({likelihood})."
        messagebox.showinfo("Prediction Result", result)

        # Clear the inputs after prediction
        for entry in entries:
            entry.delete(0, tk.END)
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")

# Create the Tkinter UI
root = tk.Tk()
root.title("Diabetes Prediction")

# Input fields with labels and help buttons
labels = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "Family History", "Age"
]

entries = []
for i, label_text in enumerate(labels):
    label = tk.Label(root, text=label_text)
    label.grid(row=i, column=0, padx=5, pady=5)

    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=5, pady=5)
    entries.append(entry)

    help_button = tk.Button(root, text="?", command=lambda t=label_text: show_help(t))
    help_button.grid(row=i, column=2, padx=5, pady=5)

(entry_pregnancies, entry_glucose, entry_bloodpressure, entry_skinthickness,
 entry_insulin, entry_bmi, entry_family_history, entry_age) = entries

# Predict button
button_predict = tk.Button(root, text="Predict", command=predict_diabetes)
button_predict.grid(row=len(labels), columnspan=3, pady=10)

# Start the Tkinter event loop
root.mainloop()
