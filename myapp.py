# app.py -- Advanced Spend Analyzer (Full upgraded)
import streamlit as st
import pandas as pd
import numpy as np
import re
import dateparser
import matplotlib.pyplot as plt
import altair as alt
from io import BytesIO
from datetime import datetime, timedelta, date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
import sqlite3
from fuzzywuzzy import process

# ----------------------------
# CONFIG / DEMO LOGO (local)
# ----------------------------
DEMO_LOGO_PATH = "/mnt/data/5d9fb76c-4f44-4f0e-8c47-af21f3860187.png"

st.set_page_config(page_title="Advanced Spend Analyzer", layout="wide", initial_sidebar_state="expanded")
st.title("💸 Advanced Monthly Spend Analyzer — Full Feature App")

# ----------------------------
# SAMPLE TRAIN DATA (offline)
# ----------------------------
SAMPLE_TRAIN = [
    ("Ordered pizza for Rs 250", "Food"),
    ("Paid at momos stall Rs 120", "Food"),
    ("Burger King payment Rs 200", "Food"),
    ("Dinner at food court Rs 450", "Food"),
    ("Watched movie at PVR Rs 350", "Entertainment"),
    ("PVR ticket booking Rs 400", "Entertainment"),
    ("Metro recharge Rs 150", "Travel"),
    ("Paid Uber Rs 220", "Travel"),
    ("Train ticket Rs 90", "Travel"),
    ("Bought fruits banana apple Rs 200", "Groceries"),
    ("Snacks tea samosa Rs 70", "Food"),
    ("Online shopping Rs 600 flipkart", "Shopping"),
    ("Electricity bill Rs 1500 paid", "Bills"),
    ("Gas bill Rs 400 paid", "Bills"),
    ("Mobile recharge Rs 199", "Recharge"),
    ("Rs 1200 electricity bill paid on 12-12-2024", "Bills"),
    ("Zomato order of Rs 180 completed yesterday", "Food"),
]

train_texts, train_labels = zip(*SAMPLE_TRAIN)
offline_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=500))
])
offline_pipeline.fit(train_texts, train_labels)

# ----------------------------
# Merchant keywords & fuzzy
# ----------------------------
MERCHANT_KEYWORDS = {
    "Swiggy": ["swiggy"],
    "Zomato": ["zomato"],
    "Dominos": ["dominos", "domino"],
    "Pizza Hut": ["pizza hut", "pizzahut"],
    "Uber": ["uber"],
    "Ola": ["ola"],
    "PVR": ["pvr", "cinema"],
    "Inox": ["inox"],
    "Amazon": ["amazon"],
    "Flipkart": ["flipkart"],
    "IRCTC": ["irctc", "railway", "train"],
    "Myntra": ["myntra"],
    "DMart": ["dmart"],
    "BSES": ["bses", "electricity"],
    "Airtel": ["airtel"],
    "Jio": ["jio"]
}
MERCHANT_NAMES = list(MERCHANT_KEYWORDS.keys())

def fuzzy_merchant_guess(token, threshold=80):
    try:
        res = process.extractOne(token, MERCHANT_NAMES)
        if res:
            name, score = res
            return name if score >= threshold else None
    except Exception:
        pass
    return None


# ----------------------------
# Optional transformer model
# ----------------------------
USE_TRANSFORMER = False

@st.cache_resource
def load_transformer():
    try:
        from transformers import pipeline
        classifier = pipeline("text-classification", model="amartyadev/banking-alerts-classifier", top_k=1)
        return classifier
    except Exception:
        return None


# ----------------------------
# Keyword scoring
# ----------------------------
KEYWORD_CATEGORIES = {
    "Bills": ["electricity", "bill", "recharge", "due", "postpaid", "broadband", "dth"],
    "Food": ["zomato", "swiggy", "pizza", "burger", "momos", "restaurant", "dine", "food"],
    "Travel": ["uber", "ola", "train", "irctc", "metro", "ticket", "cab", "flight", "bus"],
    "Entertainment": ["pvr", "cinema", "movie", "bookmyshow"],
    "Shopping": ["amazon", "flipkart", "myntra", "purchase", "order", "paid to"],
    "Groceries": ["grocery", "dmart", "store", "supermarket"],
    "Recharge": ["recharge", "topup", "mobile recharge"],
    "Miscellaneous": []
}

def keyword_score_category(text):
    text = text.lower()
    scores = {k:0 for k in KEYWORD_CATEGORIES}
    for cat, keys in KEYWORD_CATEGORIES.items():
        for key in keys:
            if key in text:
                scores[cat] += 1

    if all(v == 0 for v in scores.values()):
        return None

    priority = ["Bills", "Food", "Travel", "Entertainment",
                "Shopping", "Groceries", "Recharge", "Miscellaneous"]

    best = max(scores.items(), key=lambda x: (x[1], -priority.index(x[0])))[0]
    return best


# ----------------------------
# Main classifier (Hybrid)
# ----------------------------
def classify_text_hybrid(text, use_transformer):
    kw_cat = keyword_score_category(text)
    if kw_cat:
        return kw_cat

    if use_transformer:
        classifier = load_transformer()
        if classifier:
            try:
                pred = classifier(text)[0][0]["label"]
                label = pred.lower()
                if "food" in label:
                    return "Food"
                if "bill" in label:
                    return "Bills"
                if "travel" in label or "transport" in label:
                    return "Travel"
                return pred.capitalize()
            except Exception:
                pass

    try:
        return offline_pipeline.predict([text])[0]
    except:
        return "Miscellaneous"


# ----------------------------
# Amount & date extraction
# ----------------------------
def extract_amounts(text):
    text_low = text.replace("₹", " rs ").lower()
    matches = re.findall(r"(?:rs|inr|₹)\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", text_low)

    if not matches:
        matches = re.findall(r"\b([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)\b", text_low.replace(",", ""))

    amounts = []
    for m in matches:
        try:
            val = float(m.replace(",", ""))
            if 0 < val < 1e9:
                amounts.append(int(val))
        except:
            continue

    return amounts

def choose_amount(text, strategy="largest"):
    amts = extract_amounts(text)
    if not amts:
        return 0
    return max(amts) if strategy == "largest" else amts[0]


def extract_date(text):
    parsed = dateparser.parse(
        text,
        settings={'PREFER_DATES_FROM': 'past', 'RELATIVE_BASE': datetime.now()}
    )
    if parsed:
        return parsed.date()

    m = re.search(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b", text)
    if m:
        parsed = dateparser.parse(m.group(1))
        if parsed:
            return parsed.date()

    t = text.lower()
    now_date = datetime.now().date()
    if "yesterday" in t:
        return now_date - timedelta(days=1)
    if "today" in t:
        return now_date

    return now_date   # default


# ----------------------------
# Merchant detection
# ----------------------------
def detect_merchant(text):
    t = text.lower()
    for name, keys in MERCHANT_KEYWORDS.items():
        for k in keys:
            if k in t:
                return name

    tokens = re.findall(r"[A-Za-z0-9]{3,}", text)
    for tok in tokens[::-1]:
        guess = fuzzy_merchant_guess(tok)
        if guess:
            return guess

    return "Unknown"


# ----------------------------
# Full message analyzer
# ----------------------------
def analyze_message_full(text, amount_strategy="largest", use_transformer=False):
    category = classify_text_hybrid(text, use_transformer)
    merchant = detect_merchant(text)

    merchant_map = {
        "Swiggy": "Food", "Zomato": "Food", "Dominos": "Food", "Pizza Hut":"Food",
        "Uber":"Travel", "Ola":"Travel", "IRCTC":"Travel", "PVR":"Entertainment",
        "Inox":"Entertainment", "Amazon":"Shopping", "Flipkart":"Shopping",
        "DMart":"Groceries", "BSES":"Bills", "Airtel":"Bills", "Jio":"Bills"
    }

    if merchant in merchant_map:
        category = merchant_map[merchant]

    amount = choose_amount(text, strategy=amount_strategy)
    d = extract_date(text)
    d_str = d.strftime("%Y-%m-%d")

    return {
        "Message": text,
        "Category": category,
        "Merchant": merchant,
        "Amount": amount,
        "Date": d_str
    }


# ----------------------------
# DB Save
# ----------------------------
DB_PATH = "spend_analyzer.db"

def save_to_sqlite(df, table_name="transactions"):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table_name, conn, if_exists="append", index=False)
    conn.close()


# ----------------------------
# Prediction of next day spend
# ----------------------------
def predict_next(df):
    if df.empty:
        return None

    df2 = df.copy()
    df2["Date_parsed"] = pd.to_datetime(df2["Date"]).dt.date
    daily = df2.groupby("Date_parsed")["Amount"].sum().reset_index()

    if len(daily) < 3:
        return None

    daily = daily.reset_index(drop=True)
    daily["DayIndex"] = daily.index

    X = daily[["DayIndex"]]
    y = daily["Amount"]

    lr = LinearRegression()
    lr.fit(X, y)

    pred = lr.predict([[len(daily)]])[0]
    return max(0.0, float(pred))


# ----------------------------
# Calendar heatmap generator
# ----------------------------
def calendar_heatmap_df(df, month_str):
    try:
        year, month = map(int, month_str.split("-"))
        start = date(year, month, 1)

        if month == 12:
            end = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(year, month + 1, 1) - timedelta(days=1)

        rng = pd.date_range(start, end)
        days = pd.DataFrame({"Date": rng})
        days["DateOnly"] = days["Date"].dt.date

        df2 = df.copy()
        df2["Date_parsed"] = pd.to_datetime(df2["Date"]).dt.date

        df2 = df2[(df2["Date_parsed"] >= start) & (df2["Date_parsed"] <= end)]
        agg = df2.groupby("Date_parsed")["Amount"].sum().reset_index()
        agg = agg.rename(columns={"Date_parsed": "DateOnly", "Amount": "Amount"})

        merged = days.merge(agg, on="DateOnly", how="left").fillna(0)

        merged["Week"] = merged["Date"].dt.isocalendar().week.astype(int)
        merged["Weekday"] = merged["Date"].dt.weekday

        first_week = merged.iloc[0]["Week"]
        merged["WeekIdx"] = merged["Week"] - first_week
        merged.loc[merged["WeekIdx"] < 0, "WeekIdx"] += merged["Week"].max()

        merged["WeekIdx"] = merged["WeekIdx"].astype(int)
        return merged

    except Exception as e:
        st.error(f"Calendar processing error: {e}")
        return pd.DataFrame()


# ----------------------------
# Sidebar UI
# ----------------------------
st.sidebar.header("Options")
input_mode = st.sidebar.radio("Input mode", ["Manual paste", "Upload file (.txt/.csv)"])
amount_strategy = st.sidebar.selectbox("If multiple amounts, pick:", ["largest", "first"])

try:
    classifier = load_transformer()
    initial_transformer_value = classifier is not None
except:
    initial_transformer_value = False

use_transformer_toggle = st.sidebar.checkbox("Try transformer banking model", value=initial_transformer_value)
save_db = st.sidebar.checkbox("Save analysis to SQLite", value=False)
show_logo = st.sidebar.checkbox("Show demo merchant logo", value=True)
download_png = st.sidebar.checkbox("Enable PNG export for charts", value=True)

# Default sample messages
default_messages = f"""
Watched movie at PVR Rs 350 on {datetime.now().strftime('%d-%m-%Y')}
Snacks tea samosa Rs 70 yesterday
Online shopping Rs 600 flipkart on {(datetime.now() - timedelta(days=2)).strftime('%d-%m-%Y')}
Mobile recharge Rs 199 on {(datetime.now() - timedelta(days=10)).strftime('%d-%m-%Y')}
Paid Dominos Rs 450
Rs 1200 electricity bill paid on 12-11-2025
"""

uploaded_df = None
user_text = ""

if input_mode == "Manual paste":
    user_text = st.text_area(
        "Paste SMS messages (one per line):",
        height=220,
        value=default_messages.strip(),
        placeholder="Zomato order of Rs 180 completed yesterday\nRs 1200 electricity bill paid on 12-12-2024"
    )
else:
    uploaded_file = st.file_uploader("Upload .txt or .csv", type=["txt", "csv"])
    if uploaded_file:
        if uploaded_file.name.endswith(".txt"):
            user_text = uploaded_file.getvalue().decode("utf-8")
        else:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                if "message" not in uploaded_df.columns:
                    st.error("CSV must contain a 'message' column.")
                    uploaded_df = None
            except Exception as e:
                st.error("Failed reading CSV: " + str(e))
                uploaded_df = None

# ----------------------------
# MAIN ANALYSIS BUTTON
# ----------------------------
if st.button("Analyze"):
    if uploaded_df is not None:
        messages = uploaded_df["message"].dropna().astype(str).tolist()
    else:
        messages = [x.strip() for x in user_text.split("\n") if x.strip()]

    if not messages:
        st.warning("No messages provided.")
    else:
        # Process messages
        rows = [
            analyze_message_full(m, amount_strategy, use_transformer_toggle)
            for m in messages
        ]

        df = pd.DataFrame(rows)
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
        df["DateObj"] = pd.to_datetime(df["Date"])
        df["Month"] = df["DateObj"].dt.to_period("M").astype(str)

        # Show logo
        if show_logo:
            try:
                st.image(DEMO_LOGO_PATH, width=120)
            except:
                st.write("Demo logo not found.")

        # ---------------------------------------------------
        # DATA TABLE
        # ---------------------------------------------------
        st.subheader("📄 Extracted Transactions")
        st.dataframe(df)

        # ---------------------------------------------------
        # CATEGORY SUMMARY
        # ---------------------------------------------------
        st.subheader("📊 Spend by Category")
        cat_sum = df.groupby("Category")["Amount"].sum().reset_index()

        bar_chart = alt.Chart(cat_sum).mark_bar().encode(
            x="Category",
            y="Amount"
        )
        st.altair_chart(bar_chart, use_container_width=True)

        # ---------------------------------------------------
        # MERCHANT SUMMARY
        # ---------------------------------------------------
        st.subheader("🏪 Spend by Merchant")
        merch_sum = df.groupby("Merchant")["Amount"].sum().reset_index()

        merch_chart = alt.Chart(merch_sum).mark_bar().encode(
            x="Merchant",
            y="Amount"
        )
        st.altair_chart(merch_chart, use_container_width=True)

        # ---------------------------------------------------
        # MONTHLY TREND
        # ---------------------------------------------------
        st.subheader("📅 Monthly Trend")
        month_sum = df.groupby("Month")["Amount"].sum().reset_index()

        line = alt.Chart(month_sum).mark_line(point=True).encode(
            x="Month",
            y="Amount"
        )
        st.altair_chart(line, use_container_width=True)

        # ---------------------------------------------------
        # CALENDAR HEATMAP
        # ---------------------------------------------------
        st.subheader("🔥 Calendar Heatmap")

        month_selected = st.selectbox("Select Month", sorted(df["Month"].unique()))
        cal_df = calendar_heatmap_df(df, month_selected)

        if not cal_df.empty:
            heatmap = alt.Chart(cal_df).mark_rect().encode(
                x="WeekIdx:O",
                y="Weekday:O",
                color="Amount:Q",
                tooltip=["Date:T", "Amount:Q"]
            )
            st.altair_chart(heatmap, use_container_width=True)

        # ---------------------------------------------------
        # NEXT DAY PREDICTION
        # ---------------------------------------------------
        st.subheader("🤖 Predicted Spend for Next Day")
        pred = predict_next(df)
        if pred is None:
            st.info("Not enough data to predict.")
        else:
            st.success(f"Estimated next-day spend: **₹{pred:.2f}**")

        # ---------------------------------------------------
        # SAVE TO DB
        # ---------------------------------------------------
        if save_db:
            save_to_sqlite(df)
            st.success("Saved to SQLite database.")

