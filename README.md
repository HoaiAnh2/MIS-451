# MIS-451

## E-Commerce Shipment Delay Prediction via Classification (MIS-451)
A machine learning classification project that helps an e-commerce business reduce late deliveries by predicting whether an order will arrive on time before it is dispatched.

### Business Problem
E-commerce logistics teams struggle to proactively identify at-risk orders before delays occur, resulting in poor customer experience, high support call volumes, and reactive resource allocation. By training on historical shipment data, a classification model can flag high-risk orders at the moment they are created — enabling the operations team to prioritize handling, coordinate warehouse capacity, and notify customers in advance.

### Objectives
1. Conduct comprehensive exploratory analysis to understand variable distributions and identify factors most strongly associated with delivery delays
2. Build and compare four classification models (KNN, SVM, Logistic Regression, ANN) using consistent evaluation methodology
3. Select the best-performing model and translate results into concrete operational recommendations

### Dataset
- Source: E-Commerce Shipping Dataset (Kaggle)
- Size: 10,999 records
- Inputs: warehouse block, shipment mode, customer care calls, customer rating, product cost, prior purchases, product importance, discount offered, product weight
- Target: Reached.on.Time_Y.N — 0 = on time, 1 = late (≈60/40 imbalance)

### Workflow (Methods)
EDA
- Univariate analysis: countplots and KDE histograms for all features
- Bivariate analysis: categorical vs. target (countplot), numerical vs. target (density plot), scatter/strip plots
- Key findings: Discount_offered and Weight_in_gms showed the clearest separation between classes

Data Cleaning
- No missing values or duplicate rows detected — no imputation required
- Outliers in Prior_purchases and Discount_offered retained as they reflect genuine business events (loyal buyers, flash sales)

Preprocessing Pipeline
- Train-test split performed before any transformation (80/20, stratified, random_state=99)
- One-Hot Encoding for 3 categorical variables → 11 binary columns (fit on train only)
- StandardScaler normalization (fit on train only, transform applied to test)

Modeling (Classification)
- Logistic Regression = linear baseline
- KNN -instance-based, hyperparameter k tuned via GridSearchCV
- SVM - RBF kernel, effective on scaled multi-dimensional data
- ANN - architecture: Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.2) → Dense(1, Sigmoid), Adam optimizer, EarlyStopping

Evaluation
- Primary metric: F1-macro (chosen over accuracy due to 60/40 class imbalance)
- 10-Fold Cross-Validation on training set for all models
- Final evaluation on held-out test set (2,200 samples)

### Key Results (Summary)
* **Best selected model:** Artificial Neural Network (ANN).
* **Test performance:** Achieved an overall accuracy of 67.41% on the test set.
* The model demonstrated high cost-efficiency with a Precision of 85.06% for the 'Delayed Delivery' class, meaning out of 100 alerts generated, 85 are accurate.
* The Recall for delayed orders was 55.06%, successfully preventing 32.9% of total test order delays.

### Business Insights & Recommendations
1. Deploy a real-time alert system, integrate the ANN model into the OMS so every new order receives an automatic delay-risk score; high-risk orders are escalated in the processing queue immediately 
2. Align promotional campaigns with logistics capacity, Discount_offered is the strongest predictor of late delivery; Marketing and Logistics must coordinate before flash sales to pre-scale warehouse and transport capacity 
3. Use call volume as an early warning signal, high Customer_care_calls correlates with delay risk; equip the support team with order-risk dashboards to triage proactively 
4. Threshold tuning, lower the classification threshold from 0.5 to 0.35–0.40 to improve Recall (currently 55.1%) at the cost of more false alarms, calibrated to the FP/FN cost trade-off 
5. Quarterly model retraining, retrain periodically as delivery network and customer behavior evolve; expand features with geographic and seasonal data to close coverage gaps 

### Repository Structure
* `Project_451_Group.ipynb` — Jupyter Notebook (EDA, preprocessing, modeling)
* `Group_MIS 451_Report.docx` — Full project report
* `MIS451_GroupLazy.pdf` — Presentation slides
* `README.md` — Project overview

### How to Run
1. Open `Project_451_Group.ipynb`.
2. Install requirements (typical): `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
3. Ensure TensorFlow/Keras is installed for the ANN model.
4. Run all cells sequentially. The pipeline handles data splitting, one-hot encoding, standard scaling, and model evaluation automatically.
### Team
* **Lê Anh Khương:** Model Development and Evaluation.
* **Phạm Thúy Huyền:** Data Processing and Transformation.
* **Võ Thị Hoài Anh:** Data Overview and Preprocessing.
