
`Flower_price_2010-2025` is the processed daily input used by the updated
XGBoost workflow. It has `DATE` and `PRICES` columns and is CSV-formatted despite
having no `.csv` extension. The workflow validates dates/prices, sorts rows
chronologically, and rejects duplicate dates or non-positive prices.
