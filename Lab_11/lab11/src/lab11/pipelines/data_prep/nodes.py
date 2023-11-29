"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.14
"""

import pandas as pd


def _is_true(x: pd.Series) -> pd.Series:
    return x == "t"


def _parse_percentage(x: pd.Series) -> pd.Series:
    x = x.str.replace("%", "")
    x = x.astype(float) / 100
    return x


def _parse_money(x: pd.Series) -> pd.Series:
    x = x.str.replace("$", "").str.replace(",", "")
    x = x.astype(float)
    return x


def get_data() -> list:
    # Define las URLs directas a los archivos en bruto en el repositorio de GitHub
    url_companies = 'https://github.com/MDS7202/lab_11/raw/53c308159b801e5cbdd34bbac0985e1aadfd8895/companies.csv'
    url_shuttles = 'https://github.com/MDS7202/lab_11/raw/53c308159b801e5cbdd34bbac0985e1aadfd8895/shuttles.xlsx'
    url_reviews = 'https://raw.githubusercontent.com/MDS7202/lab_11/53c308159b801e5cbdd34bbac0985e1aadfd8895/reviews.csv'
    
    # Usa pandas para leer los archivos
    companies = pd.read_csv(url_companies)
    shuttles = pd.read_excel(url_shuttles)
    reviews = pd.read_csv(url_reviews)
    
    # Retorna un diccionario con los DataFrames
    return [companies, shuttles, reviews]



def preprocess_companies(companies: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for companies.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data, with `company_rating` converted to a float and
        `iata_approved` converted to boolean.
    """
    companies["iata_approved"] = _is_true(companies["iata_approved"])
    companies["company_rating"] = _parse_percentage(companies["company_rating"])
    return companies


def preprocess_shuttles(shuttles: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for shuttles.

    Args:
        shuttles: Raw data.
    Returns:
        Preprocessed data, with `price` converted to a float and `d_check_complete`,
        `moon_clearance_complete` converted to boolean.
    """
    shuttles["d_check_complete"] = _is_true(shuttles["d_check_complete"])
    shuttles["moon_clearance_complete"] = _is_true(shuttles["moon_clearance_complete"])
    shuttles["price"] = _parse_money(shuttles["price"])
    return shuttles


def create_model_input_table(
    shuttles: pd.DataFrame, companies: pd.DataFrame, reviews: pd.DataFrame
) -> pd.DataFrame:
    """Combines all data to create a model input table.

    Args:
        shuttles: Preprocessed data for shuttles.
        companies: Preprocessed data for companies.
        reviews: Raw data for reviews.
    Returns:
        Model input table.

    """
    rated_shuttles = shuttles.merge(reviews, left_on="id", right_on="shuttle_id")
    model_input_table = rated_shuttles.merge(
        companies, left_on="company_id", right_on="id"
    )
    model_input_table = model_input_table.dropna()

    model_input_table = model_input_table[:10000]  # for simplicity

    features = [
        "engines",
        "passenger_capacity",
        "crew",
        "d_check_complete",
        "moon_clearance_complete",
        "iata_approved",
        "company_rating",
        "review_scores_rating",
        "price",
    ]

    model_input_table = model_input_table[features]

    return model_input_table

