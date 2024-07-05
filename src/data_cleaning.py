import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import logging

class DataStrategy:
    def handle_data(self, data):
        raise NotImplementedError("This method should be overridden by subclasses")

class DataPreProcessStrategy(DataStrategy):
    """This class is used to preprocess the given dataset"""

    def __init__(self, encoder=None):
        self.encoder = encoder

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            print("Column Names Before Preprocessing:", data.columns)
            data = data.drop(["EmployeeCount", "EmployeeNumber", "StandardHours"], axis=1)

            if 'Attrition' in data.columns:
                print("Attrition column found in data.")
            else:
                print("Attrition column not found in data.")

            data["Attrition"] = data["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)
            data["Over18"] = data["Over18"].apply(lambda x: 1 if x == "Yes" else 0)
            data["OverTime"] = data["OverTime"].apply(lambda x: 1 if x == "Yes" else 0)

            # Extract categorical variables
            cat = data[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]

            # Perform one-hot encoding on categorical variables
            onehot = OneHotEncoder()
            cat_encoded = onehot.fit_transform(cat).toarray()

            # Convert cat_encoded to DataFrame
            cat_df = pd.DataFrame(cat_encoded, columns=onehot.get_feature_names_out(cat.columns))

            # Extract numerical variables
            numerical = data[['Age', 'Attrition', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']]

            # Concatenate cat_df and numerical
            data = pd.concat([cat_df, numerical.reset_index(drop=True)], axis=1)

            print("Column Names After Preprocessing:", data.columns)
            print("Preprocessed Data:")
            print(data.head())
            return data
        except Exception as e:
            logging.error(f"Error in preprocessing the data: {e}")
            raise e

# Main script to read data from CSV, preprocess, and output the results
if __name__ == "__main__":
    # Path to the CSV file
    data_path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    
    # Load data from CSV file
    data = pd.read_csv(data_path)
    
    # Create an instance of the DataPreProcessStrategy
    preprocessor = DataPreProcessStrategy()
    
    # Preprocess the data
    preprocessed_data = preprocessor.handle_data(data)
    
    # Output the preprocessed data (you can save it to a new CSV if needed)
    output_path = "data/preprocessed_data.csv"
    preprocessed_data.to_csv(output_path, index=False)
    
    print(f"Preprocessed data saved to {output_path}")
