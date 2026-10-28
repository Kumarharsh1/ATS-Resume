import os
from databricks import sql

host = os.environ["DATABRICKS_SERVER_HOSTNAME"]
http_path = os.environ["DATABRICKS_HTTP_PATH"]

try:
    with sql.connect(
        server_hostname=host,
        http_path=http_path,
        auth_type="databricks-oauth",
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            print("Databricks OAuth connected successfully!")
            print("Test result:", cursor.fetchone())

except Exception as exc:
    print("OAuth connection failed:", exc)