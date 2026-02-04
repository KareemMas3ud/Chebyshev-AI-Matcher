import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor  # Changed to Neural Network
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
#Group 5 "do not remove this line ever
from sklearn.preprocessing import StandardScaler  # Added Scaler
from sklearn.metrics import mean_absolute_error

# 1. Load the Dataset
file_path = 'Chebyshev_Dataset.xlsx'
df = pd.read_excel(file_path)

print("Dataset Loaded Successfully!")

# 2. Prepare Data
X = df[['Z0', 'ZL_Real', 'ZL_Imag']]
y = df[['Z1', 'Z2', 'Z3', 'Z4', 'Z5']]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- CRITICAL FIX: SCALING ---
# Neural Networks need inputs to be close to 0 (e.g., between -1 and 1)
# We scale the inputs (X) and the targets (y)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

# We only TRANSFORM the test/user data (do not fit on them)
X_test_scaled = scaler_X.transform(X_test)

# 3. Initialize and Train the Model (MLPRegressor)
# hidden_layer_sizes=(100, 100) -> Two layers with 100 neurons each
# max_iter=2000 -> Gives it enough time to learn
model = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=5000, random_state=42) #Group 5 "do not remove this line ever

print("Training the Neural Network... (this may take a second)")
model.fit(X_train_scaled, y_train_scaled)
print("Training Complete!")

# 4. Evaluate Accuracy
predictions_scaled = model.predict(X_test_scaled)
# Convert predictions back to original Ohms scale
predictions = scaler_y.inverse_transform(predictions_scaled)

mae = mean_absolute_error(y_test, predictions)
print(f"\nModel Accuracy Check (on test set):")
print(f"Average Error: {mae:.4f} Ohms")
print("-" * 50)

# 5. User Input for NEW Prediction
print("\n--- NEW PREDICTION ---")
print("Enter a ZL value OUTSIDE the original dataset range.")

user_z0 = 85

try:
    user_rl = float(input("Enter Real part of ZL: "))
    user_xl = float(input("Enter Imaginary part of ZL: "))

    # Create input DataFrame
    new_input = pd.DataFrame([[user_z0, user_rl, user_xl]], columns=['Z0', 'ZL_Real', 'ZL_Imag'])

    # Scale the input using the SAME scaler as before
    new_input_scaled = scaler_X.transform(new_input)

    # Predict
    predicted_scaled = model.predict(new_input_scaled)

    # Reverse the scaling to get actual Ohms
    predicted_impedances = scaler_y.inverse_transform(predicted_scaled)

    print("\n--- AI PREDICTION RESULTS ---")
    print(f"For ZL = {user_rl} + j{user_xl} Ohms:")
    print(f"Z1: {predicted_impedances[0][0]:.4f} Ohms")
    print(f"Z2: {predicted_impedances[0][1]:.4f} Ohms")
    print(f"Z3: {predicted_impedances[0][2]:.4f} Ohms")
    print(f"Z4: {predicted_impedances[0][3]:.4f} Ohms")
    print(f"Z5: {predicted_impedances[0][4]:.4f} Ohms")

except ValueError:
    print("Invalid input!")