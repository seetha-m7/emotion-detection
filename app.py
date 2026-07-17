import streamlit as st
from streamlit_autorefresh import st_autorefresh
import cv2
import pandas as pd
from deepface import DeepFace
from datetime import datetime
import os
import plotly.express as px

st.set_page_config(page_title="AI Mental Health Monitor", layout="wide")

st.title("🧠 AI Based Mental Health Emotion Detection System")

st.write("Real-Time Emotion Detection using Artificial Intelligence")

# Auto refresh every 3 seconds
st_autorefresh(interval=3000)

log_file = "emotion_log.csv"

# Create CSV file
if not os.path.exists(log_file):
    df = pd.DataFrame(columns=["emotion","time"])
    df.to_csv(log_file,index=False)

# Sidebar
st.sidebar.title("Control Panel")

run = st.sidebar.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if run:

    ret, frame = camera.read()

    if ret:

        try:

            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            emotion = result[0]['dominant_emotion']

            cv2.putText(frame, emotion, (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,(0,255,0),2)

            FRAME_WINDOW.image(frame, channels="BGR", use_container_width=True)

            df = pd.read_csv(log_file)

            new_data = {
                "emotion":emotion,
                "time":datetime.now()
            }

            df = pd.concat([df,pd.DataFrame([new_data])],ignore_index=True)

            df.to_csv(log_file,index=False)

        except:

            FRAME_WINDOW.image(frame, channels="BGR")

# Load data
df = pd.read_csv(log_file)

st.subheader("📋 Emotion Log")

st.dataframe(df.tail(10))

if len(df)>0:

    col1,col2 = st.columns(2)

    # Emotion count
    emotion_counts = df["emotion"].value_counts()

    with col1:

        st.subheader("📊 Emotion Distribution")

        fig = px.bar(
            emotion_counts,
            x=emotion_counts.index,
            y=emotion_counts.values,
            labels={'x':'Emotion','y':'Count'}
        )

        st.plotly_chart(fig,use_container_width=True)

    with col2:

        st.subheader("🥧 Emotion Pie Chart")

        pie = px.pie(
            names=emotion_counts.index,
            values=emotion_counts.values
        )

        st.plotly_chart(pie,use_container_width=True)

    # Timeline
    st.subheader("📈 Emotion Timeline")

    df["time"] = pd.to_datetime(df["time"])

    timeline = px.line(
        df,
        x="time",
        y="emotion",
        title="Emotion Changes Over Time"
    )

    st.plotly_chart(timeline,use_container_width=True)

    # Mental Health Score
    st.subheader("🧠 Mental Health Score")

    positive = ["happy","surprise"]
    negative = ["sad","angry","fear"]

    pos_count = df[df["emotion"].isin(positive)].shape[0]
    neg_count = df[df["emotion"].isin(negative)].shape[0]

    score = pos_count - neg_count

    if score > 5:

        st.success("User shows mostly positive emotional state 😊")

    elif score < -5:

        st.error("User shows signs of negative emotional state ⚠️")

    else:

        st.info("User emotional state is neutral 😐")

else:

    st.write("No emotion data yet. Start camera to collect data.")