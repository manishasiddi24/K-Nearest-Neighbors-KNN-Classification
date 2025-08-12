# K-Nearest-Neighbors-KNN-Classification
K-Nearest Neighbors (KNN) is a simple, non-parametric algorithm that classifies new data points based on the majority class among their K nearest neighbors. In this task, we used the Iris dataset and normalized features for better performance.The model was trained and tested with various values of K to compare accuracy and confusion matrices. A decision boundary visualization was created using only the first two features for simplicity. This helps understand how KNN separates classes in feature space.
 Step-by-Step Procedure
1. Load Dataset – Import the Iris dataset from Scikit-learn.
2. Normalize Features – Scale the data using StandardScaler to improve KNN performance.
3. Split Data – Divide into training and testing sets.4. Train & Evaluate Model – Use KNeighborsClassifier with different K values, check accuracy and confusion matrices.
5. Visualize Decision Boundaries – Use first two features, create meshgrid, plot classification regions.
