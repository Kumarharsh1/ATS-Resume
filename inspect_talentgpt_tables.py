
import os
from databricks import sql

TABLES = [
    "workspace.default.resumes_clean",
    "workspace.default.semantic_resume_ranking",
    "workspace.default.top_resume_candidates",
]

with sql.connect(
    server_hostname=os.environ["DATABRICKS_SERVER_HOSTNAME"],
    http_path=os.environ["DATABRICKS_HTTP_PATH"],
    auth_type="databricks-oauth",
) as connection:
    with connection.cursor() as cursor:
        for table in TABLES:
            print(f"\n{'=' * 60}")
            print(f"TABLE: {table}")
            print("=" * 60)

            try:
                cursor.execute(f"DESCRIBE TABLE {table}")
                for row in cursor.fetchall():
                    print(f"Column: {row[0]} | Type: {row[1]}")

                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                columns = [col[0] for col in cursor.description]
                print("\nSample column names:", columns)

                for row in cursor.fetchall():
                    print(row)

            except Exception as exc:
                print("Could not inspect table:", exc)
