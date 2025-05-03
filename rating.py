import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="regression_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load preprocessor dan scaler
preprocessor = joblib.load('preprocessor.pkl')
rating_scaler = joblib.load('rating_scaler.pkl')

st.title("Prediksi Rating Transaksi Supermarket")

# Input dari user
st.header("Masukkan Detail Transaksi:")

branch = st.selectbox("Cabang", ['A', 'B', 'C'])
city = st.selectbox("Kota", ['Yangon', 'Mandalay', 'Naypyitaw'])
customer_type = st.selectbox("Tipe Pelanggan", ['Member', 'Normal'])
gender = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
product_line = st.selectbox("Jenis Produk", [
    'Electronic accessories', 'Fashion accessories', 'Food and beverages',
    'Health and beauty', 'Home and lifestyle', 'Sports and travel'
])
unit_price = st.number_input("Harga Satuan", min_value=0.0, value=50.0)
quantity = st.number_input("Jumlah", min_value=1, value=1)
tax_5 = st.number_input("Pajak 5%", min_value=0.0, value=5.0)
total = st.number_input("Total", min_value=0.0, value=105.0)
date = st.date_input("Tanggal", format="YYYY-MM-DD")
time = st.time_input("Waktu")
payment = st.selectbox("Metode Pembayaran", ['Ewallet', 'Cash', 'Credit card'])
cogs = st.number_input("COGS (Harga Pokok)", min_value=0.0, value=100.0)
gross_margin_percentage = st.number_input("Persentase Margin", value=4.76)
gross_income = st.number_input("Gross Income", value=5.0)

if st.button("Prediksi Rating"):
    input_df = pd.DataFrame([{
        'Branch': branch,
        'City': city,
        'Customer type': customer_type,
        'Gender': gender,
        'Product line': product_line,
        'Unit price': unit_price,
        'Quantity': quantity,
        'Tax 5%': tax_5,
        'Total': total,
        'Date': pd.to_datetime(date).strftime('%Y-%m-%d'),
        'Time': time.strftime('%H:%M'),
        'Payment': payment,
        'cogs': cogs,
        'gross margin percentage': gross_margin_percentage,
        'gross income': gross_income
    }])

    # Preprocessing
    X_processed = preprocessor.transform(input_df).astype(np.float32)

    # TFLite requires batch input
    interpreter.set_tensor(input_details[0]['index'], X_processed)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # Inverse scaling
    y_pred = rating_scaler.inverse_transform(prediction)[0][0]

    st.success(f"Rating yang diprediksi: {y_pred:.2f}")
