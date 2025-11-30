# Quick Start Guide

## Prerequisites

1. Python 3.8+
2. PostgreSQL installed and running
3. Database `kaim` created (or update DATABASE_URL)

## Step-by-Step Setup

### 1. Clone and Setup Environment

```bash
cd customer-experience-analytics-for-fintech-apps-week2
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure Database

```bash
export DATABASE_URL="postgres://joey:yosinan@localhost:5432/kaim?schema=public"
```

Or create `.env` file (if using python-dotenv).

### 3. Update App IDs

**IMPORTANT**: Before running Task 1, update the app IDs in `src/scraper.py`:

```python
BANK_APPS = {
    "Commercial Bank of Ethiopia": {
        "app_id": "com.cbe.mobile",  # ← Update with actual app ID
        "app_name": "Commercial Bank of Ethiopia Mobile"
    },
    # ... etc
}
```

To find app IDs:
1. Go to Google Play Store
2. Search for the bank's app
3. Open the app page
4. The app ID is in the URL: `https://play.google.com/store/apps/details?id=APP_ID_HERE`

### 4. Run Tasks Sequentially

```bash
# Task 1: Scrape and preprocess reviews
python scripts/task1_scrape_reviews.py

# Task 2: Sentiment and thematic analysis
python scripts/task2_sentiment_themes.py

# Task 3: Database setup and data insertion
python scripts/task3_database_setup.py

# Task 4: Insights and visualizations
python scripts/task4_insights_visualizations.py
```

### 5. Check Outputs

- **Data**: `data/cleaned_reviews.csv`, `data/analyzed_reviews.csv`
- **Visualizations**: `figures/` directory
- **Insights**: `summary/insights_summary.txt`
- **Database**: Check with `psql` or pgAdmin

## Troubleshooting

### Scraping Fails
- Verify app IDs are correct
- Check internet connection
- Some apps may have limited reviews available

### Database Connection Fails
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Create database if needed
createdb kaim

# Test connection
psql -h localhost -U joey -d kaim
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Verify spaCy model
python -m spacy download en_core_web_sm
```

## Next Steps

1. Review generated visualizations in `figures/`
2. Read insights summary in `summary/insights_summary.txt`
3. Explore data in Jupyter notebooks (create your own in `notebooks/`)
4. Write final report based on insights

## Git Workflow

```bash
# Create branch for task 1
git checkout -b task-1

# After completing task 1
git add .
git commit -m "Task 1: Scrape and preprocess reviews"
git push origin task-1

# Create pull request and merge to main
# Repeat for tasks 2, 3, 4
```

