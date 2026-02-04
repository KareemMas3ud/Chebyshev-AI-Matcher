import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os

# PAGE CONFIGURATION
st.set_page_config(page_title="Chebyshev Transformer Project", layout="wide")

# 1. Add University Logo (BUE)
university_logo_path = "Black Text BUE Logo.png"
if os.path.exists(university_logo_path):
    st.sidebar.image(university_logo_path, use_container_width=True)
else:
    st.sidebar.warning("University logo not found. Please put 'Black Text BUE Logo.png' in the project folder.")

# Separator after logos
st.sidebar.markdown("---")

# 2. Add Department Name
st.sidebar.markdown(
    """
    <div style="text-align: center; font-weight: bold; margin-bottom: 20px;">
    Department of Electrical<br>
    Engineering and Communications
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()

# 3. Project Team Section
st.sidebar.title("🎓 Project Team")
st.sidebar.info("""
**Supervised By:**
* Prof. Hani Ghali
* TA: Malak Ibrahim

**Prepared By:**
* Kareem Mohammed (238253)
* Rawan Essam (235067)
* Kenzy Ashraf (219253)
* Jana Ahmed (219537)
""")

st.sidebar.divider()

# MAIN TITLE WITH LOGO
col1, col2 = st.columns([4, 1])

with col1:
    st.markdown("# ⚡ Chebyshev Multisection Matching Transformers")
    st.markdown("""
    **Electromagnetic Waves Project** | Faculty of Engineering (BUE)
    
    This tool uses an **AI Model (Neural Network)** to design a broadband impedance-matching network. 
    It compares the AI's instant predictions against the manual **Small Reflection Theory** calculations.
    """)

with col2:
    project_logo_path = "logo.png"
    if os.path.exists(project_logo_path):
        st.image(project_logo_path, use_container_width=True)
    else:
        st.warning("Logo not found.")
st.divider()


# 1. LOAD & TRAIN MODEL
@st.cache_resource
def train_model():
    # Load Data
    try:
        df = pd.read_excel('Chebyshev_Dataset.xlsx')
    except FileNotFoundError:
        st.error("❌ Error: 'Chebyshev_Dataset.xlsx' not found. Please put it in the same folder.")
        return None, None, None

    X = df[['Z0', 'ZL_Real', 'ZL_Imag']]
    y = df[['Z1', 'Z2', 'Z3', 'Z4', 'Z5']]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    # Train
    model = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu',
                         solver='adam', max_iter=5000, random_state=42)
    model.fit(X_train_scaled, y_train_scaled)

    return model, scaler_X, scaler_y


model, scaler_X, scaler_y = train_model()

# 2. SIDEBAR INPUTS
st.sidebar.header("🛠️ Design Parameters")
Z0 = st.sidebar.number_input("Characteristic Impedance ($Z_0$)", value=85.0)
ZL_Real = st.sidebar.number_input("Load Real Part ($R_L$)", value=300.0)
ZL_Imag = st.sidebar.number_input("Load Imaginary Part ($X_L$)", value=200.0)

# Calculate Complex Load
ZL_complex = complex(ZL_Real, ZL_Imag)
st.sidebar.markdown(f"**Target Load:** ${ZL_Real} + j{ZL_Imag} \Omega$")

# 3. AI PREDICTION
if model is not None:
    # Prepare input for AI
    user_input = pd.DataFrame([[Z0, ZL_Real, ZL_Imag]], columns=['Z0', 'ZL_Real', 'ZL_Imag'])
    user_input_scaled = scaler_X.transform(user_input)

    # Predict
    pred_scaled = model.predict(user_input_scaled)
    ai_impedances = scaler_y.inverse_transform(pred_scaled)[0]


    # 4. THEORETICAL CALCULATION (Validation)
    def calculate_manual(Z0, ZL, am=0.05, N=5):
        # Reflection Coefficient Magnitude
        Gamma_L = (ZL - Z0) / (ZL + Z0)
        Gamma_mag = abs(Gamma_L)

        # Chebyshev Constant S
        val_check = (1 / N) * np.arccosh((1 / am) * Gamma_mag)
        if ((1 / am) * Gamma_mag) < 1:
            S = 1.0
        else:
            S = np.cosh(val_check)

        # Reflection Coefficients (Symmetry for N=5)
        Gamma0 = (am / 2) * (S ** 5)
        Gamma1 = (am / 2) * (5 * (S ** 5) - 5 * (S ** 3))
        Gamma2 = (am / 2) * (10 * (S ** 5) - 15 * (S ** 3) + 5 * S)

        # Calculate Impedances (Small Reflection Theory)
        Z1_th = Z0 * np.exp(2 * Gamma0)
        Z2_th = Z1_th * np.exp(2 * Gamma1)
        Z3_th = Z2_th * np.exp(2 * Gamma2)
        Z4_th = Z3_th * np.exp(2 * Gamma2)  # Symmetric
        Z5_th = Z4_th * np.exp(2 * Gamma1)  # Symmetric

        return [Z1_th, Z2_th, Z3_th, Z4_th, Z5_th], Gamma_mag, S


    theo_impedances, Gamma_mag, S_val = calculate_manual(Z0, ZL_complex)

    # 5. DISPLAY RESULTS
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 AI Model Output")
        st.info("Characteristic Impedances Prediction")
        st.write(f"**Z1:** {ai_impedances[0]:.4f} $\Omega$")
        st.write(f"**Z2:** {ai_impedances[1]:.4f} $\Omega$")
        st.write(f"**Z3:** {ai_impedances[2]:.4f} $\Omega$")
        st.write(f"**Z4:** {ai_impedances[3]:.4f} $\Omega$")
        st.write(f"**Z5:** {ai_impedances[4]:.4f} $\Omega$")

    with col2:
        st.subheader("📐 Theoretical Output")
        st.success("Manual Calculation (Validation)")
        st.write(f"**Z1:** {theo_impedances[0]:.4f} $\Omega$")
        st.write(f"**Z2:** {theo_impedances[1]:.4f} $\Omega$")
        st.write(f"**Z3:** {theo_impedances[2]:.4f} $\Omega$")
        st.write(f"**Z4:** {theo_impedances[3]:.4f} $\Omega$")
        st.write(f"**Z5:** {theo_impedances[4]:.4f} $\Omega$")

    # 6. ERROR TABLE
    st.subheader("📊 Accuracy Verification (% Error)")
    errors = [abs((t - a) / t) * 100 for t, a in zip(theo_impedances, ai_impedances)]
    error_df = pd.DataFrame({
        "Section": ["Z1", "Z2", "Z3", "Z4", "Z5"],
        "AI Prediction": ai_impedances,
        "Manual Theory": theo_impedances,
        "% Error": errors
    })
    st.table(error_df.style.format({"AI Prediction": "{:.4f}", "Manual Theory": "{:.4f}", "% Error": "{:.4f}%"}))

    # 7. PLOTTING
    st.subheader("📈 Frequency Response (Chebyshev)")

    # Plot Setup
    theta = np.linspace(0, np.pi, 300)
    f_norm = 2 * theta / np.pi
    am = 0.05

    fig, ax = plt.subplots(figsize=(10, 4))

    # Calculate Response
    x = S_val * np.cos(theta)
    Tn = 16 * (x ** 5) - 20 * (x ** 3) + 5 * x
    Gamma_resp = am * np.abs(Tn)

    ax.plot(f_norm, Gamma_resp, label='N=5 Response', color='#1f77b4', linewidth=2.5)
    ax.axhline(am, color='red', linestyle='--', label='Ripple Limit (0.05)')

    ax.set_xlabel("Normalized Frequency ($f/f_0$)")
    ax.set_ylabel("Reflection Coefficient $|\Gamma|$")
    ax.set_title(f"Bandwidth Response for Load: {ZL_Real} + j{ZL_Imag} $\Omega$")
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.legend()
    ax.set_ylim(0, max(0.6, max(Gamma_resp) * 1.1))

    st.pyplot(fig)