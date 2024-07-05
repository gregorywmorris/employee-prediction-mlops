import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from typing import Union

class DataStrategy:
    def handle_data(self, data):
        raise NotImplementedError("This method should be overridden by subclasses")

class DataDivideStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            if 'Attrition' in data.columns:
                X = data.drop(['Attrition'], axis=1)
                Y = data['Attrition']
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
                return X_train, X_test, Y_train, Y_test
            else:
                raise ValueError("'Attrition' column not found in data.")
        except Exception as e:
            logging.error(f"Error in data handling: {str(e)}")
            raise e

if __name__ == "__main__":
    data_path = "./data/preprocessed_data.csv"
    data = pd.read_csv(data_path)
    divider = DataDivideStrategy()
    X_train, X_test, y_train, y_test = divider.handle_data(data)
    print(f"Data divided: {X_train.shape[0]} training rows, {X_test.shape[0]} testing rows")
