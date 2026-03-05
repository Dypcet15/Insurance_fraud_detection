import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class FeatureEngineering:
    def __init__(self, df):
        self.df = df
        self.encoder = OneHotEncoder(sparse=False, drop='first')
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def create_features(self):
        # Example feature creation
        self.df['total_amount'] = self.df['quantity'] * self.df['price']

    def encode_categorical_features(self, categorical_cols):
        encoded_features = self.encoder.fit_transform(self.df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_features, columns=self.encoder.get_feature_names_out(categorical_cols))
        self.df = pd.concat([self.df, encoded_df], axis=1)
        self.df.drop(categorical_cols, axis=1, inplace=True)

    def scale_features(self, numerical_cols):
        self.df[numerical_cols] = self.scaler.fit_transform(self.df[numerical_cols])

    def handle_outliers(self, numerical_cols):
        for col in numerical_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            IQR = q3 - q1
            lower_bound = q1 - 1.5 * IQR
            upper_bound = q3 + 1.5 * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]

    def process(self, categorical_cols, numerical_cols):
        self.create_features()
        self.encode_categorical_features(categorical_cols)
        self.handle_outliers(numerical_cols)
        self.scale_features(numerical_cols)
        return self.df

# Example usage:
# df = pd.read_csv('data.csv')
# feature_engineering = FeatureEngineering(df)
# processed_df = feature_engineering.process(categorical_cols=['category_col'], numerical_cols=['quantity', 'price'])