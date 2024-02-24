################################
###### Final Home Exercise #####
################################

################################
# Student ID: 209372325
# First and Last Names: Maya Raskin
################################

# In this exercise you should implement a classification pipeline which aims at predicting the amount of hours
# a worker will be absent from work based on the worker characteristics and the work day missed.
# Download the dataset from the course website, which is provided as a .csv file. The target label is 'TimeOff'.
# You are free to use any library functions from numpy, pandas and sklearn, etc...
#
# You should implement the body of the functions below. The main two points of entry to your code are DataPreprocessor class and
# the train_model function. In the '__main__' section you are provided with an example of how your submission will be evaluated. 
# You are free to change the body of the functions and classes as you like - as long as it adheres to the provided input & output structure.
# In all methods and functions the input structure and the required returned variables are explicitly stated.
# Note that in order to evaluate the generalization error, you'll need to run cross validation as we demonstrated in class,
# However!!! In the final submission your file needs to contain only the methods of DataPreprocessor and the train_model function.
# Tip: You are encouraged to run gridsearch to find the best model and hyperparameters as demonstrated in class.
#
# To make things clear: you need to experiment with the preprocessing stage and the final model that will be used to fit. To get the
# sense of how your model performs, you'll need to apply the CV approach and, quite possibly, do a grid search of the meta parameters. 
# In the end, when you think that you've achieved your best, you should make a clean - and runnable!!! - version of your insights,
# which must adhere to the api provided below. In the evaluation stage, your code will be run on the entire train data,
# and then run once on the test data.
#
# You are expected to get results between 50% and 100% accuracy on the test set.
# Of course, the test set is not provided to you. However, as previously mentioned, running cross validation
# (with enough folds) will give you a good estimation of the accuracy.
#
# Important: obtaining accuracy less than 60%, will grant you 65 points for this exercise.
# Obtaining accuracy score above 60% will grant you 75 points minimum, however, your final score
# will be according to the distribution of all submissions. Therefore, due to the competition nature of this exercise, 
# you may use any method or library that will grant you the highest score, even if not learned in class.
#
# Identical or equivalent submissions will give rise to a suspicion of plagiarism.
#
# In addition to stating your names and ID numbers in the body of this file, name the file as follows:
# ex4_FirstName_LastName.py


import numpy as np
from pandas.core.groupby import DataFrameGroupBy
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder


def load_dataset(train_csv_path: str):
    return pd.read_csv(train_csv_path, sep=',')


class DataPreprocessor(object):
    """
    This class is a mandatory API.

    The purpose of this class is to unify data preprocessing step between the training and the testing stages. 
    This may include, but not limited to, the following transformations:
    1. Filling missing (NA / nan) values
    2. Dropping non-descriptive columns
    3 ...

    The test data is unavailable when building the ML pipeline, thus it is necessary to determine the 
    preprocessing steps and values on the train set and apply them on the test set.


    *** Mandatory structure ***
    The ***fields*** are ***not*** mandatory
    The ***methods***  - "fit" and "transform" - are ***required***.

    You're more than welcome to use sklearn.pipeline for the "heavy lifting" of the preprocessing tasks, but it is not an obligation. 
    Any class that implements the methods "fit" and "transform", with the required inputs & outputs will be accepted.
    Even if "fit" performs nothing at all.
    """

    def __init__(self):
        self.transformer = None

    def display_top_10_rows(self, df: pd.DataFrame) -> None:
        print('Top 10 rows:')
        print(df.head(10))

    def where_are_the_nans(self, df: pd.DataFrame) -> None:
        cols_with_nan_values: pd.Series = df.isna().sum().loc[lambda x: x > 0]
        print("Columns with at least one Nan value:")
        print(cols_with_nan_values)

    def fit(self, dataset_df: pd.DataFrame) -> None:
        """
        Input:
        dataset_df: the training data loaded from the csv file as a dataframe containing only the features
        (not the target - see the main function).

        Output:
        None

        Functionality:
        Based on all the provided training data, this method learns with which values to fill the NA's, 
        how to scale the features, how to encode categorical variables etc.
        Handle the relevant columns and save the information needed to transform the fields in the instance state.

        """
        dataset_df.info()

        self.display_top_10_rows(dataset_df)
        self.where_are_the_nans(dataset_df)

    def convert_textual_binary_to_boolean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Prepare encoder
        l_encoder = LabelEncoder()
        l_encoder.fit(['Yes', 'No'])

        # Transform columns
        df['Bin_Smoker'] = l_encoder.transform(df['Smoker'])
        df['Bin_Drinker'] = l_encoder.transform(df['Drinker'])

        return df

    def ordinal_converter(self, df: pd.DataFrame, col: str, ctgs: list[str]) -> pd.DataFrame:
        enc = OrdinalEncoder(categories=[ctgs], handle_unknown='use_encoded_value', unknown_value=np.nan)
        df[col.replace(' ', '_') + '_Numeric'] = enc.fit_transform(df[[col]])

        return df

    def convert_age_group_to_ordinal(self, df: pd.DataFrame) -> pd.DataFrame:
        ctgs = ['Young Adult', 'Adult', 'Middle Aged', 'Senior']
        return self.ordinal_converter(df, col='Age Group', ctgs=ctgs)

    def convert_education_to_ordinal(self, df: pd.DataFrame) -> pd.DataFrame:
        ctgs = ['High school', 'Graduate', 'Postgraduate', 'Phd']
        return self.ordinal_converter(df, col='Education', ctgs=ctgs)

    def fill_nan(self, group: DataFrameGroupBy, col: str, fallback_value: any, inplace: bool):
        non_null = group[col].notnull()

        if non_null.any():
            group[col].fillna(method='bfill', inplace=inplace)

        group[col].fillna(fallback_value, inplace=inplace)

        return group

    def fill_na(self, df: pd.DataFrame) -> pd.DataFrame:
        # Fill height and weight
        non_null_weight = df['Weight'].notna()
        df['Height'].fillna(df['Weight'][non_null_weight] + 100, inplace=True)

        non_null_height = df['Height'].notna()
        df['Weight'].fillna(df['Height'][non_null_height] - 100, inplace=True)

        df = df.groupby('ID', group_keys=False).apply(
            self.fill_nan, col='Height', fallback_value=df['Height'].mean(), inplace=True
        )

        df = df.groupby('ID', group_keys=False).apply(
            self.fill_nan, col='Weight', fallback_value=df['Weight'].mean(), inplace=True
        )

        # Fill residence distance with mean value
        df = df.groupby('ID', group_keys=False).apply(
            self.fill_nan, col='Residence Distance', fallback_value=df['Residence Distance'].mean(), inplace=True
        )

        # Fill service time with mean value
        df = df.groupby('ID', group_keys=False).apply(
            self.fill_nan, col='Service time', fallback_value=df['Service time'].mean(), inplace=True
        )

        # Fill Age Group with most common
        df = df.groupby('ID', group_keys=False).apply(
            self.fill_nan, col='Age Group', fallback_value=df['Age Group'].value_counts().idxmax(), inplace=True
        )

        # Fill Education with 'High school' for young adults, and most common for others
        # TODO: Decrease dimensions
        df = df.groupby('ID', group_keys=False).apply(
            self.fill_nan, col='Education', fallback_value=df['Education'].value_counts().idxmax(), inplace=True
        )

        # Fill Son with 0 for young adults, and most common for others
        df = df.groupby('ID', group_keys=False).apply(
            self.fill_nan, col='Son', fallback_value=0, inplace=True
        )

        # Fill Drinker with most common
        df = df.groupby('ID', group_keys=False).apply(
            self.fill_nan, col='Drinker', fallback_value='No', inplace=True
        )

        df = df.groupby('ID', group_keys=False).apply(
            self.fill_nan, col='Drinker', fallback_value=df['Drinker'].value_counts().idxmax(), inplace=True
        )

        # Fill Smoker with Yes
        df = df.groupby('ID', group_keys=False).apply(
            self.fill_nan, col='Smoker', fallback_value='Yes', inplace=True
        )

        # Fill Pet with most common
        df = df.groupby('ID', group_keys=False).apply(
            self.fill_nan, col='Pet', fallback_value=df['Pet'].value_counts().idxmax(), inplace=True
        )

        # Fill Season according to month (there are no NaN values in the Month column)
        # Summer - 1
        # Winter - 2
        # Spring - 3
        # Fall - 4

        month_to_season = {
            1: 2, 2: 2, 3: 3, 4: 3, 5: 3, 6: 1, 7: 1, 8: 1, 9: 4, 10: 4, 11: 4, 12: 2
        }
        df['Season'].fillna(df['Month'].map(month_to_season), inplace=True)

        return df

    def get_important_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=['ID', 'Height', 'Weight', 'Smoker', 'Drinker', 'Education', 'Age Group', 'Reason'])

    def convert_reason(self, df: pd.DataFrame) -> pd.DataFrame:
        one_hot_encoded = pd.get_dummies(df['Reason'], prefix='Reason')
        df = pd.concat([df, one_hot_encoded], axis=1)
        return df

    def transform(self, df: pd.DataFrame):
        """
        Input:
        df:  *any* data similarly structured to the train data (dataset_df input of "fit")

        Output:
        A processed dataframe or ndarray containing only the input features (X).
        It should maintain the same row order as the input.
        Note that the labels vector (y) should not exist in the returned ndarray object or dataframe.

        Functionality:
        Based on the information learned in the "fit" method, apply the required transformations to the passed data (df)
        *** This method will be called exactly once during evaluation. See the main section for details ***

        """

        c_df = df.copy()

        # Fill N/A values
        c_df = self.fill_na(c_df)

        # Introduce new column to drop Weight and Height
        c_df['BMI'] = c_df['Weight'] / pow(c_df['Height'], 2) * 100

        # Convert Yes/No to True/False
        c_df = self.convert_textual_binary_to_boolean(c_df)

        # Convert age group to ordinal
        c_df = self.convert_age_group_to_ordinal(c_df)

        # Convert education to ordinal
        c_df = self.convert_education_to_ordinal(c_df)

        # Convert Reason using one hot
        c_df = self.convert_reason(c_df)

        # Get important features
        c_df = self.get_important_features(c_df)

        return c_df


def train_model(processed_x: pd.DataFrame, y: pd.DataFrame):
    """
    This function gets the data after the pre-processing stage  - after running DataPreprocessor.transform on it, 
    a vector of labels, and returns a trained model. 

    Input:
    processed_X (ndarray or dataframe): the data after the pre-processing stage
    y: a vector of labels

    Output:
    model: an object with a "predict" method, which accepts the ***pre-processed*** data and outputs the prediction


    """
    param_grid = {
        'n_estimators': np.arange(10, 100, 10),
        'max_depth': [8, 10],
    }

    model = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, verbose=3, cv=5)
    model.fit(processed_x, y)

    return model

def split_data(processed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = processed_df["TimeOff"]
    x = processed_df.drop(["TimeOff"], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1, stratify=y
    )

    # This splits the data into a train set, which will be used to calibrate the internal parameters of predictor, and the test set, which will be used for checking
    print("Shapes of X_train, Y_train, X_test and Y_test:")
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    return x_train, x_test, y_train, y_test

def MAMA_main():
    preprocessor = DataPreprocessor()
    train_csv_path = 'time_off_data_train.csv'
    train_dataset_df = load_dataset(train_csv_path)

    x_train = train_dataset_df.iloc[:, :-1]
    y_train = train_dataset_df['TimeOff']
    preprocessor.fit(x_train)
    model = train_model(preprocessor.transform(x_train), y_train)

    ### Evaluation Section ####
    test_csv_path = 'time_off_data_train.csv'
    # Obviously, this will be different during evaluation. For now, you can keep it to validate proper execution
    test_csv_path = train_csv_path
    test_dataset_df = load_dataset(test_csv_path)

    x_test = test_dataset_df.iloc[:, :-1]
    y_test = test_dataset_df['TimeOff']

    processed_x_test = preprocessor.transform(x_test)
    predictions = model.predict(processed_x_test)
    test_score = accuracy_score(y_test, predictions)
    print("Accuracy on test:", test_score)

    predictions = model.predict(preprocessor.transform(x_train))
    train_score = accuracy_score(y_train, predictions)
    print('Accuracy on train:', train_score)

def main():
    preprocessor = DataPreprocessor()
    train_csv_path = 'time_off_data_train.csv'
    train_dataset_df = load_dataset(train_csv_path)

    processed_def = preprocessor.transform(train_dataset_df)
    x_train, x_test, y_train, y_test = split_data(processed_def)

    preprocessor.fit(x_train)
    model = train_model(x_train, y_train)

    predictions = model.predict(x_test)
    test_score = accuracy_score(y_test, predictions)
    print("Accuracy on test:", test_score)

    predictions = model.predict(x_train)
    train_score = accuracy_score(y_train, predictions)
    print('Accuracy on train:', train_score)

if __name__ == '__main__':
    main()
