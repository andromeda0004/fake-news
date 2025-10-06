# Fake news detection

## The problem

The problem is not only hackers, going into accounts, and sending false information. The bigger problem here is what we call "Fake News". A fake are those news stories that are false: the story itself is fabricated, with no verifiable facts, sources, or quotes.

When someone (or something like a bot) impersonates someone or a reliable source to false spread information, that can also be considered as fake news. In most cases, the people creating this false information have an agenda, that can be political, economical or to change the behavior or thought about a topic.

There are countless sources of fake news nowadays, mostly coming from programmed bots, that can't get tired (they're machines hehe) and continue. to spread false information 24/7.

Serious studies in the past 5 years, have demonstrated big correlations between the spread of false information and elections, the popular opinion or feelings about different topics.

The problem is real and hard to solve because the bots are getting better are tricking us. Is not simple to detect when the information is true or not all the time, so we need better systems that help us understand the patterns of fake news to improve our social media, communication and to prevent confusion in the world.

## Purpose

In this short code , I'll explain several ways to detect fake news using collected data from different articles. But the same techniques can be applied to different scenarios. For the coders and experts, I'll explain the Python code to load, clean, and analyze data. Then we will do some machine learning models to perform a classification task (fake or not).

## The Data

The data comes from Kaggle, you can download it here:

https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

There are two files, one for real news and one for fake news (both in English) with a total of 23481 "fake" tweets and 21417 "real" articles.

## 🚀 Streamlit Web Application

### Local Setup & Running

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. **Open browser at:** `http://localhost:8501`

### Deploy on Streamlit Cloud

1. **Push your changes to GitHub:**
   ```bash
   git add .
   git commit -m "Add Streamlit deployment"
   git push origin master
   ```

2. **Deploy:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `app.py`
   - Click "Deploy"

### ⚠️ Important Notes

- The model was trained on **political news from 2016-2017**
- Works best on political articles
- May not generalize well to science, tech, or entertainment news
- Use political news examples for best results
- See `POLITICAL_NEWS_EXAMPLES.md` for test examples

### Model Performance

- **Accuracy:** 99.77% (on test data)
- **Model:** Random Forest Classifier
- **Features:** TF-IDF vectorized text
- **Training Data:** 44,898 political articles


