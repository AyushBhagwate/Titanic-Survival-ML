# Lib 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

# Pipeline
def get_pipeline(num_col, cat_col):

    num_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessing = ColumnTransformer([
        ('num', num_pipeline, num_col),
        ('cat', cat_pipeline, cat_col)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessing),
        ('yeo', PowerTransformer(method='yeo-johnson')) # Fixed skew
    ])
    
    return pipeline


