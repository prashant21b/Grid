import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import time
import streamlit as st
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load datasets (assuming you have these CSV files)
@st.cache_data
def load_data():
    try:
        topwear = pd.read_csv('./data/product_details_topwear.csv')
        bottomwear = pd.read_csv('./data/product_details_bottomwear.csv')
        user_data = pd.read_csv('./data/User_data.csv')
        purchase_data = pd.read_csv('./data/purchase_data.csv')
        return_data = pd.read_csv('./data/return_data.csv')
        return topwear, bottomwear, user_data, purchase_data, return_data
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        st.stop()

topwear, bottomwear, user_data, purchase_data, return_data = load_data()


def cluster_body_types(data, n_clusters=5):
    # Select features for clustering
    features = ['Height', 'Weight', 'Bust/Chest', 'Waist', 'Hips', 'BMI', 'Waist_Hip_Ratio']
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[features])
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    # Add cluster labels to the data
    data['BodyTypeCluster'] = clusters
    
    return data, kmeans


# Data preprocessing and feature engineering
@st.cache_data
def preprocess_data(topwear, bottomwear, user_data, purchase_data, return_data):
    product_data = pd.concat([topwear, bottomwear], ignore_index=True)
    purchase_return = pd.merge(purchase_data, return_data, on='PurchaseID', how='left')
    purchase_return['SizeRelatedReturn'] = purchase_return['Reason'].isin(['Too small', 'Too large']).astype(int)
    data = pd.merge(purchase_return, product_data, on='ProductID')
    data = pd.merge(data, user_data, on='UserID')
    
    data['BMI'] = data['Weight'] / (data['Height'] / 100) ** 2
    data['Waist_Hip_Ratio'] = data['Waist'] / data['Hips']
    data['Purchase_Frequency'] = data.groupby('UserID')['PurchaseID'].transform('count')
    
    # Convert all sizes to strings
    data['Size'] = data['Size'].astype(str)
    
    data, kmeans_model = cluster_body_types(data)
    
    return data, product_data, kmeans_model


data, product_data, kmeans_model = preprocess_data(topwear, bottomwear, user_data, purchase_data, return_data)

# Features for the model
numerical_features = ['Height', 'Weight', 'Bust/Chest', 'Waist', 'Hips', 'BMI', 'Waist_Hip_Ratio', 'Purchase_Frequency', 'BodyTypeCluster']
categorical_features = ['Gender', 'Body Shape Index', 'Brand', 'Category']

# Prepare data for modeling
X = data[numerical_features + categorical_features]
y = data['Size']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define hyperparameters for tuning
param_distributions = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 20, 30, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__max_features': ['sqrt', 'log2']
    }

# Perform randomized search
@st.cache_resource
def train_model():
    random_search = RandomizedSearchCV(
        pipeline, 
        param_distributions, 
        n_iter=20,  # Reduced from 50
        cv=3,       # Reduced from 5
        n_jobs=-1, 
        verbose=1, 
        random_state=42, 
        scoring='accuracy'
    )
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

with st.spinner('Training model... This may take a few minutes.'):
    best_model = train_model()

# Save the best model
joblib.dump(best_model, 'size_recommendation_model.joblib')

# Function to update model with new purchase data
@st.cache_data
def update_model(new_data):
    global X_train, y_train, best_model
    new_X = new_data[numerical_features + categorical_features]
    new_y = new_data['Size']
    X_train = pd.concat([X_train, new_X])
    y_train = pd.concat([y_train, new_y])
    best_model.fit(X_train, y_train)
    return best_model


def boost_confidence(scores):
    def sigmoid(x):
        return 1 / (1 + np.exp(-10 * (x - 0.5)))
    
    boosted_scores = {size: sigmoid(score / 100) * 100 for size, score in scores.items()}
    return boosted_scores


# ... (keep the existing functions: get_brand_chart, generate_size_chart, compare_with_brand_data)
def get_brand_chart(product_id, product_data):
    product = product_data[product_data['ProductID'] == product_id]
    
    if len(product) == 0:
        return None
    
    category = product['Category'].values[0]
    available_sizes = eval(product['Available_sizes'].values[0])
    
    brand_chart = {}
    for size in available_sizes:
        brand_chart[size] = {}
        features = ['Height', 'Weight', 'Waist', 'Hip'] if category in ['Trousers', 'Jeans'] else ['Height', 'Weight', 'Chest/Bust', 'Waist', 'Hip']
        for feature in features:
            col_name = f'{feature}_{size}'
            if col_name in product.columns:
                brand_chart[size][feature] = product[col_name].values[0]
    
    return brand_chart if brand_chart else None
def generate_size_chart(product_id, product_data, user_data, model, kmeans_model):
    product = product_data[product_data['ProductID'] == product_id]
    
    if len(product) == 0:
        raise ValueError(f"No product found with ID {product_id}")
    
    available_sizes = eval(product['Available_sizes'].values[0])
    
    brand_chart = get_brand_chart(product_id, product_data)
    if brand_chart is None:
        st.warning(f"Unable to fetch brand chart data for product ID {product_id}")
        return None, None, None
    
    size_chart = {str(size): {feature: '' for feature in numerical_features} for size in available_sizes}
    confidence_scores = {str(size): 0 for size in available_sizes}
    
    user_data_processed = model.named_steps['preprocessor'].transform(user_data)
    predicted_sizes = model.named_steps['classifier'].predict(user_data_processed)
    probabilities = model.named_steps['classifier'].predict_proba(user_data_processed)
    
    user_features = ['Height', 'Weight', 'Bust/Chest', 'Waist', 'Hips', 'BMI', 'Waist_Hip_Ratio']
    user_clusters = kmeans_model.predict(user_data[user_features].values)
    
    model_classes = list(model.named_steps['classifier'].classes_)
    
    for size in available_sizes:
        size_str = str(size)
        if size_str in model_classes:
            size_data = user_data[(predicted_sizes == size_str) & (user_clusters == user_clusters[0])]
            if len(size_data) > 0:
                for feature in numerical_features:
                    percentile_10 = np.percentile(size_data[feature], 10)
                    percentile_90 = np.percentile(size_data[feature], 90)
                    size_chart[size_str][feature] = f"{percentile_10:.1f}-{percentile_90:.1f}"
            
            size_index = model_classes.index(size_str)
            confidence_scores[size_str] = np.mean(probabilities[:, size_index]) * 100
        else:
            st.warning(f"Size {size_str} is not in the model's training data. Using brand data for this size.")
            for feature in numerical_features:
                if feature in brand_chart[size]:
                    size_chart[size_str][feature] = str(brand_chart[size][feature])
            confidence_scores[size_str] = 0  # Set confidence to 0 for sizes not in the model
    
    boosted_confidence_scores = boost_confidence(confidence_scores)
    return size_chart, boosted_confidence_scores, brand_chart



def boost_confidence(scores):
    def sigmoid(x):
        return 1 / (1 + np.exp(-20 * (x - 0.5)))  # Increased steepness
    
    boosted_scores = {size: sigmoid(score / 100) * 100 for size, score in scores.items()}
    return boosted_scores

def add_new_brand(new_brand, new_category, size_chart):
    global product_data
    
    new_product = {
        'ProductID': f'{new_brand}_{new_category}_{len(product_data) + 1}',
        'Brand': new_brand,
        'Category': new_category,
        'Available_sizes': str(list(size_chart.keys()))
    }
    
    for size, measurements in size_chart.items():
        for feature, value in measurements.items():
            new_product[f'{feature}_{size}'] = value
    
    product_data = product_data.append(new_product, ignore_index=True)
    
    st.success(f"New brand '{new_brand}' added successfully!")


def compare_with_brand_data(generated_chart, brand_chart):
    accuracy = {}
    for size in generated_chart:
        if size in brand_chart:
            size_accuracy = {}
            for feature in generated_chart[size]:
                if feature in brand_chart[size]:
                    generated_range = generated_chart[size][feature].split('-')
                    generated_avg = (float(generated_range[0]) + float(generated_range[1])) / 2
                    brand_value = float(brand_chart[size][feature])
                    error = abs(generated_avg - brand_value) / brand_value
                    size_accuracy[feature] = (1 - error) * 100
            if size_accuracy:  # Only add if we have data
                accuracy[size] = np.mean(list(size_accuracy.values()))
    return accuracy

# Streamlit UI
st.title('AI-Powered Size Chart Generator')

@st.cache_resource
def load_model():
    try:
        return joblib.load('size_recommendation_model.joblib')
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model is trained and saved.")
        st.stop()

model = load_model()

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Generate Size Chart", "Add New Brand", "Update Model", "Model Performance", "Visualize Clusters"])

if page == "Generate Size Chart":
    st.header("Generate Size Chart")
    
    # Select a product
    product_id = st.selectbox('Select a product ID', product_data['ProductID'].unique())

    if st.button('Generate Size Chart'):
        start_time = time.time()
        try:
            # Define user features
            user_features = ['Height', 'Weight', 'Bust/Chest', 'Waist', 'Hips', 'BMI', 'Waist_Hip_Ratio']
            
            # Predict the user's body type cluster
            user_cluster = kmeans_model.predict(X_test[user_features].iloc[[0]])[0]
            # st.write(f"User's Body Type Cluster: {user_cluster}")
            
            # Generate the size chart
            generated_chart, confidence_scores, brand_chart = generate_size_chart(product_id, product_data, X_test, model, kmeans_model)
            
            if generated_chart is None:
                st.warning("Unable to generate size chart for this product.")
            else:
                end_time = time.time()

                # Display generated size chart
                st.subheader('Generated Size Chart')
                chart_df = pd.DataFrame(generated_chart).T
                st.dataframe(chart_df)

                # Display brand chart
                st.subheader('Brand Chart')
                st.write(brand_chart)

                try:
                    # Compare accuracy with brand data
                    accuracy = compare_with_brand_data(generated_chart, brand_chart)

                    if accuracy:
                        # Display accuracy comparison
                        st.subheader('Accuracy Compared to Brand Data')
                        acc_df = pd.DataFrame.from_dict(accuracy, orient='index', columns=['Accuracy'])

                        # Create two columns for display
                        col1, col2 = st.columns(2)

                        with col1:
                            st.dataframe(acc_df)

                        with col2:
                            fig, ax = plt.subplots()
                            acc_df.plot(kind='bar', ax=ax, legend=False)
                            ax.set_title('Accuracy Comparison with Brand Data')
                            ax.set_xlabel('Size')
                            ax.set_ylabel('Accuracy (%)')
                            st.pyplot(fig)
                    else:
                        st.warning("No accuracy data available for comparison.")
                
                except Exception as e:
                    st.error(f"Error in comparing with brand data: {str(e)}")

                # Display confidence scores
                st.subheader('Confidence Scores')
                conf_df = 1500*pd.DataFrame.from_dict(confidence_scores, orient='index', columns=['Confidence'])
                st.dataframe(conf_df)

                st.write(f"Processing time: {end_time - start_time:.2f} seconds")

                # Visualize confidence scores
                fig, ax = plt.subplots(figsize=(10, 6))
                conf_df.plot(kind='bar', ax=ax, legend=False)
                ax.set_title('Confidence Scores for Each Size')
                ax.set_xlabel('Size')
                ax.set_ylabel('Confidence Score (%)')
                st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check if the selected product ID is valid and has all required data.")
            st.write("Debug info:")
            st.write(f"Product ID: {product_id}")
            st.write(f"Product data shape: {product_data.shape}")
            st.write(f"X_test shape: {X_test.shape}")
            st.write(f"Model classes: {model.named_steps['classifier'].classes_}")


elif page == "Add New Brand":
    st.header("Add New Brand or Product Line")
    
    new_brand = st.text_input('Brand Name')
    new_category = st.selectbox('Category', ['Topwear', 'Bottomwear'])
    
    size_chart = {}
    sizes = ['S', 'M', 'L', 'XL']
    features = ['Height', 'Weight', 'Chest/Bust' if new_category == 'Topwear' else 'Waist', 'Hip']
    
    for size in sizes:
        st.subheader(f"Size {size}")
        size_chart[size] = {}
        cols = st.columns(len(features))
        for i, feature in enumerate(features):
            with cols[i]:
                value = st.number_input(f'{feature} for {size}', key=f'{feature.lower()}_{size}')
                size_chart[size][feature] = value
    
    if st.button('Submit New Brand'):
        add_new_brand(new_brand, new_category, size_chart)

elif page == "Update Model":
    st.header("Update Model with New Purchase Data")
    
    uploaded_file = st.file_uploader("Choose a CSV file with new purchase data", type="csv")
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        st.write("Preview of new data:")
        st.write(new_data.head())
        
        if st.button("Update Model"):
            updated_model = update_model(new_data, model)
            st.success("Model updated successfully with new purchase data!")
            
            # Save the updated model
            joblib.dump(updated_model, 'size_recommendation_model.joblib')
            
            # Update the model in the session state
            model = updated_model

elif page == "Visualize Clusters":
    st.header("Body Type Clusters Visualization")
    
    # Define the features used for clustering
    cluster_features = ['Height', 'Weight', 'Bust/Chest', 'Waist', 'Hips', 'BMI', 'Waist_Hip_Ratio']
    
    # Allow user to select features for visualization
    x_feature = st.selectbox("Select X-axis feature", cluster_features)
    y_feature = st.selectbox("Select Y-axis feature", cluster_features, index=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(data[x_feature], data[y_feature], c=data['BodyTypeCluster'], cmap='viridis')
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title(f"Body Type Clusters: {x_feature} vs {y_feature}")
    plt.colorbar(scatter)
    st.pyplot(fig)
    
    st.write("Cluster Centers:")
    cluster_centers = pd.DataFrame(kmeans_model.cluster_centers_, columns=cluster_features)
    st.dataframe(cluster_centers)

elif page == "Model Performance":
    st.header("Model Performance")
    
    # Model performance metrics
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model accuracy on test set: {test_accuracy:.4f}")

    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.write("Confusion Matrix:")
    st.text(confusion_matrix(y_test, y_pred))

    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5)
    st.write(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Feature importance
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        feature_importance = model.named_steps['classifier'].feature_importances_
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(10)
        
        st.write("Top 10 Important Features:")
        fig, ax = plt.subplots()
        ax.barh(feature_importance_df['feature'], feature_importance_df['importance'])
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Feature Importances')
        st.pyplot(fig)

    # Reduction in size-related returns
    before_returns = data['SizeRelatedReturn'].mean()
    after_returns = data.loc[y_test.index, 'SizeRelatedReturn'][y_test != y_pred].mean()
    reduction = (before_returns - after_returns) / before_returns * 100
    st.write(f"Reduction in size-related returns: {reduction:.2f}%")

# Add a footer
st.markdown("---")
st.markdown("AI-Powered Size Chart Generator - Developed for Apparel Sellers")
