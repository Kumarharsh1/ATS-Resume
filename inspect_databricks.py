
import os
from databricks import sql


def main():
    hostname = os.environ["DATABRICKS_SERVER_HOSTNAME"]
    http_path = os.environ["DATABRICKS_HTTP_PATH"]

    try:
        with sql.connect(
            server_hostname=hostname,
            http_path=http_path,
            auth_type="databricks-oauth",
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SHOW CATALOGS")
                catalogs = [row[0] for row in cursor.fetchall()]

                print("\nConnected to Databricks successfully!")
                print("\nAvailable catalogs:")

                for catalog in catalogs:
                    print(f"\n{'=' * 12} CATALOG: {catalog} {'=' * 12}")

                    try:
                        cursor.execute(f"SHOW SCHEMAS IN `{catalog}`")
                        schemas = [row[0] for row in cursor.fetchall()]

                        for schema in schemas:
                            print(f"\nSchema: {catalog}.{schema}")

                            try:
                                cursor.execute(
                                    f"SHOW TABLES IN `{catalog}`.`{schema}`"
                                )
                                tables = cursor.fetchall()

                                if not tables:
                                    print("  No tables found.")

                                for table in tables:
                                    print(f"  Table: {table[1]}")

                            except Exception as exc:
                                print(f"  Could not list tables: {exc}")

                    except Exception as exc:
                        print(f"Could not inspect catalog: {exc}")

    except Exception as exc:
        print("\nDatabricks connection failed:", exc)


if __name__ == "__main__":
    main()
