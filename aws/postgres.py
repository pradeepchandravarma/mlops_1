

import boto3
import json
from botocore.exceptions import ClientError

def get_secret():
    secret_name = "pradeep-mlops-postgres"
    region_name = "eu-west-2"

    session = boto3.session.Session()
    client = session.client(
        service_name="secretsmanager",
        region_name=region_name
    )

    try:
        response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise e

    return json.loads(response["SecretString"])


from sqlalchemy import create_engine

secret = get_secret()

engine = create_engine(
    f"postgresql+psycopg2://{secret['username']}:{secret['password']}"
    f"@{secret['host']}:{secret['port']}/{secret['dbname']}"
)
import pandas as pd

df = pd.read_csv("Data/Student_Performance.csv")
print(df.head())
print(df.dtypes)

from sqlalchemy import text

with engine.connect() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS mlops"))
    conn.commit()
df.to_sql(
    name="performance_metrics",
    con=engine,
    schema="mlops",
    if_exists="replace",   # use "append" if table already exists
    index=False
)
query = "SELECT * FROM mlops.performance_metrics LIMIT 5"
print(pd.read_sql(query, engine))

