import pandas as pd
from databricks import sql
import os
from dotenv import load_dotenv
load_dotenv('.env')
from src.constants import DEFAULT_CATALOG, DEFAULT_SCHEMA, DEFAULT_TABLE, DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN


class Source_Connectors:
    """
    Connector to fetch table data from Databricks SQL and return as pandas.DataFrame.
    Does NOT write to disk.
    """


    def __init__(self,
                 host: str = DATABRICKS_HOST,
                 http_path: str = DATABRICKS_HTTP_PATH,
                 token: str = DATABRICKS_TOKEN,
                 catalog: str = DEFAULT_CATALOG,
                 schema: str = DEFAULT_SCHEMA,
                 table: str = DEFAULT_TABLE):
        self.host = host
        self.http_path = http_path
        self.token = token
        self.full_table_name = f"{catalog}.{schema}.{table}"


        if not all([self.host, self.http_path, self.token]):
            missing = []
            if not self.host:
                missing.append("DATABRICKS_HOST")
            if not self.http_path:
                missing.append("DATABRICKS_HTTP_PATH")
            if not self.token:
                missing.append("DATABRICKS_TOKEN")
            raise EnvironmentError(f"Missing Databricks credentials: {', '.join(missing)}")


    def fetch_dataframe(self, sql_query: str = None) -> pd.DataFrame:
        """
        Execute a SQL query against Databricks and return the result as a pandas DataFrame.
        If sql_query is None, it will SELECT * from the configured table.
        """
        query = sql_query or f"SELECT * FROM {self.full_table_name};"
        try:
            with sql.connect(
                server_hostname=self.host,
                http_path=self.http_path,
                access_token=self.token
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    arrow_table = cursor.fetchall_arrow()
                    df = arrow_table.to_pandas()
                return df
        except Exception as e:
            # re-raise so upstream code can handle/log appropriately
            raise

