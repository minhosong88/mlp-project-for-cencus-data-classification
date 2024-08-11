import pandas as pd
from sklearn.preprocessing import LabelEncoder, PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


def load_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(axis=0, how='any', inplace=True)
    return df


def preprocess_data(df, n_components=3):
    # Encode 'State' column and drop 'County'
    label_encode = LabelEncoder()
    df['State'] = label_encode.fit_transform(df['State'])
    df.drop('County', axis=1, inplace=True)

    # Define target and features
    X = df.drop(columns='ChildPoverty')
    y = df['ChildPoverty']

    # Normalize the data
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    pipeline = Pipeline([
        ('scl', StandardScaler()),
        ('pca', PCA(n_components=n_components)),
    ])
    preprocessor = ColumnTransformer([
        ('numeric', pipeline, numeric_features),
    ])
    X_preprocessed = preprocessor.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed, y, test_size=0.2, random_state=1)

    return X_train, X_test, y_train, y_test


def balance_and_split(df):
    # Balance and split the target variable
    thresholds = [0, 6.2, 16.3, 31.6, df['ChildPoverty'].max()]
    df['ChildPoverty'] = pd.cut(
        df['ChildPoverty'], bins=thresholds, labels=False, include_lowest=True)
    return df


def plot_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df['ChildPoverty'], bins=30, color='skyblue', edgecolor='black')
    plt.axvline(df['ChildPoverty'].mean(), color='red',
                linestyle='dashed', linewidth=1, label='Mean')
    plt.axvline(df['ChildPoverty'].median(), color='green',
                linestyle='dashed', linewidth=1, label='Median')
    plt.xticks(range(0, 101, 25))
    plt.xlabel('Child Poverty Level')
    plt.ylabel('Frequency')
    plt.title('Distribution of Child Poverty Levels')
    plt.legend()
    plt.grid(True)
    plt.show()
