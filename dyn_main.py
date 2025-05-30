import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

with open("dyn_br.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = df.dropna(subset=['text', 'dynasty'])
df['text'] = df['text'].astype(str)
df['dynasty'] = df['dynasty'].astype(str)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier()
}

dynasty_results = []
for model_name, model in models.items():
    accs = []
    for _ in range(2):
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['dynasty'], test_size=0.2, random_state=np.random.randint(10000))
        pipeline = make_pipeline(TfidfVectorizer(), model)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        accs.append(accuracy_score(y_test, preds))
    dynasty_results.append((model_name, round(np.mean(accs), 4)))

df = df.dropna(subset=['period'])
df['period'] = df['period'].astype(str)

period_avg_acc = {}
for model_name, model in models.items():
    acc_sum = 0
    dynasty_count = 0
    for dynasty, subdf in df.groupby('dynasty'):
        if subdf['period'].nunique() < 2:
            continue
        accs = []
        for _ in range(2):
            X_train, X_test, y_train, y_test = train_test_split(subdf['text'], subdf['period'], test_size=0.2, random_state=np.random.randint(10000))
            pipeline = make_pipeline(TfidfVectorizer(), model)
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            accs.append(accuracy_score(y_test, preds))
        acc_sum += np.mean(accs)
        dynasty_count += 1
    period_avg_acc[model_name] = round(acc_sum / dynasty_count, 4) if dynasty_count > 0 else 0.0

print("\nClassifier\tDynasty\tPeriod (avg)")
for model_name, dyn_acc in dynasty_results:
    print(f"{model_name}\t{dyn_acc:.4f}\t{period_avg_acc.get(model_name, 0.0):.4f}")
