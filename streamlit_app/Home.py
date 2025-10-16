import streamlit as st
import os

# --- Always first ---
st.set_page_config(page_title="Micro Tools for Trading", layout="wide")

# --- App Title ---
st.title("Micro Tools for Trading")
st.write("Welcome! This is the first version of your trading ML tool suite.")

# --- Sidebar navigation ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Home", "MLR with NIFTY50"])

# --- Page router logic ---
if app_mode == "Home":
    st.subheader("🏠 Home")
    st.write("You are on the Home page.")
    st.markdown("""
    - Use the sidebar to navigate to your trading tools.
    - Currently available: **MLR with NIFTY50**
    """)

elif app_mode == "MLR with NIFTY50":
    st.subheader("📈 MLR with NIFTY50")

    # Build path to your script dynamically (works regardless of where Streamlit runs)
    file_path = os.path.join(r"D:\mini_tradeApp\micro-tools-trading\streamlit_app", "MLR_with_nifty50.py")


    # Safety check before executing
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
            # Execute the file content directly
            exec(code, globals())
    else:
        st.error(f"File not found: {file_path}")

    st.markdown("---")
    st.caption("You are in MLR with NIFTY50 page.")
