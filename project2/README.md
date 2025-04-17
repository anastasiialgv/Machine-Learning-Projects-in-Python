This project demonstrates the construction and evaluation of a simple ensemble classifier using the scikit-learn library on a synthetic two‑moon dataset. First, we generate 10,000 points arranged in two interleaving half‑circles using make_moons(n_samples=10000, noise=0.4). We then split the dataset into training (80%) and test (20%) subsets via train_test_split, ensuring reproducibility with a fixed random_state.

Next, we instantiate and train three base classifiers on the training set:

Logistic Regression, a linear model that outputs class probabilities via the logistic (sigmoid) function;

Support Vector Machine (SVM) with an RBF kernel (enabled by default in scikit-learn’s SVC), which finds the maximal‑margin hyperplane in a transformed feature space;

Random Forest, an ensemble of decision trees built on bootstrap samples and random feature subsets to reduce variance and overfitting.

We combine these three classifiers into a soft‑voting ensemble (VotingClassifier) that averages their predicted probabilities before making a final class decision. To assess performance, we report the ensemble’s accuracy on both the training and test sets. Finally, we visualize the model’s decision boundary by evaluating the ensemble over a dense grid spanning the feature space and plotting filled contours alongside the true test points. This illustrates how the voting ensemble blends the strengths of individual classifiers to form a smoother, more robust decision surface.
