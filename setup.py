from setuptools import setup, find_packages

setup(
    name="laliga_predictions",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
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
) 