# 라이브러리 및 데이터 불러오기

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns


wine = load_wine()

# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다.
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.

# 데이터 로딩
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 하이퍼파라미터 설정
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5],                 # ← 변경된 부분
    'min_samples_split': [2, 5, 10],          # ← 변경된 부분
    'min_samples_leaf': [1, 2, 4]
}

# 4. 모델 정의 및 GridSearchCV 적용
dt_model = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=dt_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

#  5. 최적 모델 평가
best_dt = grid_search.best_estimator_
y_pred_dt = best_dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("Best Hyperparameters (DT):", grid_search.best_params_)
print("Best Accuracy (DT):", dt_accuracy)
print("Classification Report (DT):\n", classification_report(y_test, y_pred_dt))

#  6. Feature Importance 시각화
importances = best_dt.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=importance_df, x='Feature', y='Importance')
plt.title(" Decision Tree Feature Importance")
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()