# Customer Experience Analytics for Fintech Apps

A Real-World Data Engineering Challenge: Scraping, Analyzing, and Visualizing Google Play Store Reviews

## Challenge Overview

This week's challenge centers on analyzing customer satisfaction with mobile banking apps by collecting and processing user reviews from the Google Play Store for three Ethiopian banks:

- **Commercial Bank of Ethiopia (CBE)**
- **Bank of Abyssinia (BOA)**
- **Dashen Bank**

You'll scrape app reviews, analyze sentiments and themes, and visualize insights as a Data Analyst at Omega Consultancy, a firm advising banks.

Building on Week 1's foundational skills, this week introduces web scraping, thematic NLP analysis, and basic database engineering.

## Business Objective

Omega Consultancy is supporting banks to improve their mobile apps to enhance customer retention and satisfaction. Your role as a Data Analyst is to:

1. Scrape user reviews from the Google Play Store
2. Analyze sentiment (positive/negative/neutral) and extract themes (e.g., "bugs", "UI")
3. Identify satisfaction drivers (e.g., speed) and pain points (e.g., crashes)
4. Store cleaned review data in a Postgres database
5. Deliver a report with visualizations and actionable recommendations

## Project Structure

```
customer-experience-analytics-for-fintech-apps-week2/
├── .gitignore
├── requirements.txt
├── README.md
├── database_schema.sql
├── data/
│   ├── raw_reviews.csv
│   ├── cleaned_reviews.csv
│   ├── analyzed_reviews.csv
│   └── keywords_by_bank.csv
├── notebooks/
│   └── __init__.py
├── scripts/
│   ├── __init__.py
│   ├── task1_scrape_reviews.py
│   ├── task2_sentiment_themes.py
│   ├── task3_database_setup.py
│   └── task4_insights_visualizations.py
├── src/
│   ├── __init__.py
│   ├── scraper.py
│   ├── sentiment_analysis.py
│   ├── thematic_analysis.py
│   ├── database.py
│   ├── insights.py
│   └── visualizations.py
├── tests/
│   └── __init__.py
├── figures/
│   └── (generated visualizations)
└── summary/
    └── insights_summary.txt
```

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 4. Configure Database

Set the `DATABASE_URL` environment variable:

```bash
export DATABASE_URL="postgres://joey:yosinan@localhost:5432/kaim?schema=public"
```

Or create a `.env` file:

```
DATABASE_URL=postgres://joey:yosinan@localhost:5432/kaim?schema=public
```

### 5. Update App IDs

Before running Task 1, update the app IDs in `src/scraper.py` with the actual Google Play Store app IDs for the three banks.

## Tasks Overview

### Task 1: Data Collection and Preprocessing

Scrape reviews from the Google Play Store, preprocess them, and save as CSV.

**Run:**
```bash
python scripts/task1_scrape_reviews.py
```

**Output:**
- `data/raw_reviews.csv` - Raw scraped data
- `data/cleaned_reviews.csv` - Preprocessed data

**KPIs:**
- 1,200+ reviews collected with <5% missing data
- Clean CSV dataset
- Organized Git repo with clear commits

### Task 2: Sentiment and Thematic Analysis

Quantify review sentiment and identify themes to uncover satisfaction drivers and pain points.

**Run:**
```bash
python scripts/task2_sentiment_themes.py
```

**Output:**
- `data/analyzed_reviews.csv` - Reviews with sentiment and theme labels
- `data/keywords_by_bank.csv` - Extracted keywords per bank

**KPIs:**
- Sentiment scores for 90%+ reviews
- 3+ themes per bank with examples
- Modular pipeline code

### Task 3: Store Cleaned Data in PostgreSQL

Design and implement a relational database in PostgreSQL to store processed review data.

**Run:**
```bash
python scripts/task3_database_setup.py
```

**Output:**
- PostgreSQL database with `banks` and `reviews` tables
- `database_schema.sql` - Database schema file

**KPIs:**
- Working database connection + insert script
- Tables populated with >1,000 review entries
- SQL dump or schema file committed to GitHub

### Task 4: Insights and Recommendations

Derive insights from sentiment and themes, visualize results, and recommend app improvements.

**Run:**
```bash
python scripts/task4_insights_visualizations.py
```

**Output:**
- `figures/` - All visualizations (sentiment, ratings, themes, word clouds)
- `summary/insights_summary.txt` - Comprehensive insights report

**KPIs:**
- 2+ drivers/pain points with evidence
- Clear, labeled visualizations
- Practical recommendations

## Key Dates

- **Introduction**: 10:30 AM UTC, Wednesday, 26 Nov 2025
- **Interim Submission**: 8:00 PM UTC, Sunday, 30 Nov 2025
- **Final Submission**: 8:00 PM UTC, Tuesday, 02 Dec 2025

## Scenarios

### Scenario 1: Retaining Users
CBE has a 4.2, BOA 3.4, and Dashen 4.1 star rating. Users complain about slow loading during transfers. Analyze if this is a broader issue and suggest areas for app investigation.

### Scenario 2: Enhancing Features
Extract desired features (e.g., transfer, fingerprint login, faster loading times) through keyword and theme extraction. Recommend how each bank can stay competitive.

### Scenario 3: Managing Complaints
Cluster and track complaints (e.g., "login error") to guide AI chatbot integration and faster support resolution strategies.

## Methodology

### Data Collection
- Uses `google-play-scraper` library to collect reviews
- Targets minimum 400+ reviews per bank (1,200 total)
- Handles rate limiting and error cases

### Sentiment Analysis
- Primary method: VADER (fast, rule-based)
- Alternative: DistilBERT (transformers-based, more accurate but slower)
- Aggregates sentiment by bank and rating

### Thematic Analysis
- TF-IDF for keyword extraction
- Manual theme assignment based on keyword matching
- Optional: LDA/NMF topic modeling for advanced analysis

### Database Schema
- **Banks Table**: `bank_id`, `bank_name`, `app_name`
- **Reviews Table**: `review_id`, `bank_id`, `review_text`, `rating`, `review_date`, `sentiment_label`, `sentiment_score`, `theme`, `source`

### Visualizations
- Sentiment distribution by bank
- Rating distribution and averages
- Theme distribution
- Sentiment trends over time
- Word clouds per bank
- Comprehensive bank comparison dashboard

## Testing

Run unit tests (when implemented):

```bash
pytest tests/
```

## Git Workflow

1. Create feature branches for each task:
   ```bash
   git checkout -b task-1
   # ... work on task 1 ...
   git commit -m "Task 1: Scrape and preprocess reviews"
   git push origin task-1
   ```

2. Merge via pull request to main branch

3. Repeat for tasks 2, 3, and 4

## References

### Web Scraping
- [google-play-scraper](https://github.com/JoMingyu/google-play-scraper)

### Sentiment Analysis
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- [TextBlob](https://textblob.readthedocs.io/)

### NLP & Thematic Analysis
- [spaCy](https://spacy.io/)
- [scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Topic Modeling (LDA/NMF)](https://scikit-learn.org/stable/modules/decomposition.html)

### Database
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [psycopg2](https://www.psycopg.org/docs/)
- [SQLAlchemy](https://docs.sqlalchemy.org/)

### Testing
- [pytest](https://docs.pytest.org/)

## Notes

- **App IDs**: Update the app IDs in `src/scraper.py` with actual Google Play Store app IDs
- **Database**: Ensure PostgreSQL is running and the database `kaim` exists
- **Rate Limiting**: Scraping includes delays to respect API limits
- **Data Quality**: Scripts include data quality checks and validation

## Troubleshooting

### Scraping Issues
- Verify app IDs are correct
- Check internet connection
- Increase delay between requests if rate-limited

### Database Connection Issues
- Verify PostgreSQL is running: `sudo systemctl status postgresql`
- Check database exists: `psql -l`
- Verify connection string format

### spaCy Model Not Found
```bash
python -m spacy download en_core_web_sm
```

### Memory Issues with Large Datasets
- Process reviews in batches
- Use VADER instead of DistilBERT for faster processing

## License

This project is part of the KAIM Data Engineering Challenge.
