import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

log_clf = LogisticRegression()
svm_clf = SVC(probability=True)
rf_clf  = RandomForestClassifier(random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('svm', svm_clf), ('rf', rf_clf)],
    voting='soft'
)

voting_clf.fit(X_train, y_train)

train_accuracy = voting_clf.score(X_train, y_train)
test_accuracy  = voting_clf.score(X_test, y_test)
print(f"Training Accuracy: {train_accuracy:.3f}")
print(f"Test Accuracy:     {test_accuracy:.3f}")


x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

Z = voting_clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', s=10)
plt.title('Decision Boundary of VotingClassifier')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
plt.show()
