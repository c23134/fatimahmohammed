
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="تصنيف إشارات القلب - فاطمة", layout="centered")

st.title("🩺 تصنيف إشارات القلب باستخدام النموذج الذكي")
st.markdown("### ارفعي ملف إشارات (.csv) ليتم تصنيفه تلقائيًا باستخدام النموذج.")

uploaded_file = st.file_uploader("اختر ملف CSV يحتوي على إشارة القلب", type=["csv"])

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

def preprocess_signal(df):
    # التأكد من أن البيانات عبارة عن صف واحد فقط (180 نقطة مثلًا)
    signal = df.values.flatten()
    signal = signal[:180] if len(signal) > 180 else np.pad(signal, (0, 180 - len(signal)))
    signal = signal.reshape(1, 180, 1)
    return signal

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, header=None)
        st.success("تم تحميل الإشارة بنجاح ✅")
        st.line_chart(df.T)

        signal = preprocess_signal(df)
        model = load_model()

        prediction = model.predict(signal)
        label_map = {0: "N - طبيعي", 1: "S - Supraventricular", 2: "V - Ventricular", 3: "F - Fusion", 4: "Q - Unknown"}
        predicted_class = np.argmax(prediction)

        st.markdown(f"### 🔍 التنبؤ: {label_map[predicted_class]}")
        st.write("نسبة الثقة:", f"{np.max(prediction)*100:.2f}%")
    except Exception as e:
        st.error(f"حدث خطأ أثناء المعالجة: {e}")
else:
    st.info("يرجى رفع ملف لتبدأ عملية التصنيف.")
