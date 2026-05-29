from setuptools import setup, find_packages

setup(
    name="laliga_predictions",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "pyspark>=3.5.0,<4.0.0",
        "numpy",
        "scikit-learn",
        "xgboost",
        "mlflow",
        "prefect",
        "hyperopt",
        "dagshub",
        "requests",
        "lxml",
        "unidecode",
        "python-dotenv",
        "pytest",
    ],
    python_requires=">=3.11,<3.12",
) 
