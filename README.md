# AI-Powered Size Chart Generator

This project is an AI-powered size recommendation system designed for apparel sellers. It helps generate accurate size charts based on user measurements, purchase history, and return/exchange data, significantly reducing size-related returns.

## Demo 
  https://apparelseller.streamlit.app/
## Features
- *Size Chart Generation*: Predicts the best size for users based on their body measurements, past purchases, and returns.
- *Brand Comparison*: Compares the generated size chart with the official brand size chart for accuracy.
- *Clustered Body Types*: Uses KMeans clustering to group users into body types, improving size prediction accuracy.
- *Confidence Scores*: Assigns confidence scores to each size recommendation and boosts them using a sigmoid function.
- *Model Training & Updates*: Supports model training and updates with new data using RandomForestClassifier and RandomizedSearchCV for hyperparameter tuning.
- *Interactive UI*: Built with Streamlit, providing a user-friendly interface for generating size charts and managing brand data.

## Installation

1. Clone the repository:
    bash
    git clone [https://github.com/yourusername/size-chart-generator.git](https://github.com/prashant21b/Grid)
    cd size-chart-generator
    

2. Install dependencies:
    bash
    pip install -r requirements.txt
    

3. Run the Streamlit app:
    bash
    streamlit run app.py
    

## Project Structure
- app.py: Main Streamlit app that handles the UI and size chart generation.
- model_training.py: Contains the model training and update logic.
- data/: Directory containing the datasets (product_details_topwear.csv, product_details_bottomwear.csv, user_data.csv, purchase_data.csv, return_data.csv).
- size_recommendation_model.joblib: Trained RandomForest model for size recommendation.
- README.md: Project documentation.

## Usage

1. *Generate Size Chart*: Select a product ID from the available list to generate a size chart for that product based on user measurements and compare it with the brand chart.
2. *Add New Brand*: Add a new brand or product line with size-specific measurements, and update the brand data.
3. *Update Model*: Use new purchase data to retrain the model and improve its accuracy.
4. *Model Performance*: View the model's accuracy and performance metrics, including confidence scores.
5. *Visualize Clusters*: Visualize the clustering of user body types.

## Data

- product_details_topwear.csv: Contains details of topwear products (ProductID, Category, Brand, Available Sizes, Size-specific measurements).
- product_details_bottomwear.csv: Contains details of bottomwear products.
- user_data.csv: Contains user measurements (Height, Weight, Bust/Chest, Waist, Hips).
- purchase_data.csv: Contains user purchase history.
- return_data.csv: Contains reasons for returns (size-related or others).

## Model Details

- *RandomForestClassifier*: A random forest model is used for size prediction. The model is tuned using RandomizedSearchCV with cross-validation.
- *Clustering*: KMeans is used to group users into body type clusters, which improves size recommendation accuracy.
- *Sigmoid Boosting*: Confidence scores for size recommendations are boosted using a sigmoid function.

## Adding a New Brand

To add a new brand or product line, go to the "Add New Brand" page in the app. Enter the brand name, category, and size-specific measurements. The new brand will be added to the product_data DataFrame and stored for future use.


Efforts are underway to improve accuracy and confidence scores through model tuning and data enrichment.

## Future Enhancements
- Integrate feedback loop from return data to fine-tune size predictions.
- Expand the dataset with additional brands and categories.
- Add support for international sizing standards.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
