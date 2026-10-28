
import os
from databricks import sql

try:
    with sql.connect(
        server_hostname=os.environ["DATABRICKS_SERVER_HOSTNAME"],
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
        access_token=os.environ["DATABRICKS_TOKEN"],
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            print("Databricks connected successfully!")
            print("Test result:", cursor.fetchone())

except Exception as e:
    print("Connection failed:", e)
