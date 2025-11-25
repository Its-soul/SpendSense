# 💸 SpendSense — Intelligent SMS-Based Monthly Expense Analyzer

SpendSense is a smart **NLP-powered expense analyzer** that extracts spending patterns directly from SMS messages.  
It uses **Hybrid NLP (keyword + fuzzy match)** and optional **Transformer-based classification** to detect:

- 🛒 Categories (Food, Travel, Movies, Shopping, Bills, etc.)
- 🏪 Merchants (Swiggy, Zomato, PVR, Uber, Amazon, etc.)
- 💰 Spend Amounts (₹ formats, multi-amount messages)
- 📅 Dates (multiple SMS formats)
- 📊 Monthly, Weekly & Daily spending trends  
- 🔮 Predictive analytics (Next-month spend forecast)

This project runs on **Streamlit**, giving you a beautiful dashboard interface.

<img width="1831" height="857" alt="image" src="https://github.com/user-attachments/assets/6fa5bddf-a1c1-4ac3-946b-f796dde3fae6" />
example of calender 
<img width="1860" height="425" alt="image" src="https://github.com/user-attachments/assets/0d706660-df53-4fd1-9a5c-981e35d16632" />



---

## 🚀 Features

### 🔍 NLP-Powered Extraction
- Keyword-based merchant detection  
- Fuzzy matching for misspellings (e.g., "swigy", "uber rde")  
- Date parsing from SMS  
- Multi-amount detection (largest or custom rule)  

### 🧠 Optional ML Mode
- Uses Transformers (BERT-based model) for text classification  
- Auto fallback to offline TF-IDF model if internet unavailable

### 📁 File Support
- Upload `.txt` or `.csv` containing SMS messages  
- Direct multi-line input in the text box  

### 📊 Visualizations
- Bar chart  
- Pie chart  
- Line plot (daily + monthly timeline)  
- Download chart as PNG  



### 1️⃣ Clone the repo
```sh
git clone https://github.com/Its-soul/SpendSense.git
cd SpendSense
