from cryptography.fernet import Fernet
import pyodbc
from typing import Dict, Any, Tuple, Optional, Union
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

# Fallback MySQL connector
try:
    import mysql.connector
    HAVE_MYSQL = True
except ImportError:
    HAVE_MYSQL = False

class DatabaseConnection:
    def __init__(self):
        self.key = b'zfis_enruV2dLQxC9OE_ajTb5PP2uqZxahoqncm4RFQ='
        
    def decrypt_password(self, encrypted_password: str) -> str:
        """Decrypt password using stored key"""
        try:
            print(f"Attempting to decrypt password...")
            if not encrypted_password:
                print("No password provided")
                return None
            
            f = Fernet(self.key)
            decrypted = f.decrypt(encrypted_password.encode()).decode()
            print("Password decrypted successfully")
            return decrypted
        
        except Exception as e:
            print(f"Error decrypting password: {str(e)}")
            # Return None instead of raising error
            return None
    
    def get_credentials(self, datamartid: str) -> Optional[Dict[str, Any]]:
        """Get database credentials for given datamartid. For local testing, returns direct table access."""
        try:
            print("\n=== Using local MySQL credentials ===")
            load_dotenv()
            
            # For local testing, just return direct table access
            credentials = [{
                'table_name': datamartid,  # Use the datamartid as table name
                'database': os.getenv('DB_NAME'),
                'username': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'source_type': 'Table'
            }]
            
            print(f"✓ Using table: {datamartid}")
            return credentials
            
        except Exception as e:
            print(f"\n✗ Error getting credentials: {str(e)}")
            raise
    
    def connect_to_database(self, credentials: Dict[str, Any]) -> Tuple[Any, Any]:
        """Connect to database using provided credentials. Tries ODBC first, falls back to native connector."""
        try:
            # First try ODBC
            try:
                connection_string = (
                    f"Driver={{MySQL ODBC 8.0 Unicode Driver}};"
                    f"Server={os.getenv('DB_SERVER')};"
                    f"Database={credentials['database']};"
                    f"UID={os.getenv('DB_USER')};"
                    f"PWD={os.getenv('DB_PASSWORD')};"
                )
                
                connection = pyodbc.connect(connection_string)
                cursor = connection.cursor()
                print("✓ Connected via ODBC")
                return connection, cursor
                
            except Exception as e:
                print(f"ODBC connection failed: {str(e)}")
                if not HAVE_MYSQL:
                    raise Exception("ODBC failed and mysql-connector-python not installed. Try: pip install mysql-connector-python")
                
                # Fall back to native MySQL connector
                print("Trying native MySQL connector...")
                connection = mysql.connector.connect(
                    host=os.getenv('DB_SERVER'),
                    user=os.getenv('DB_USER'),
                    password=os.getenv('DB_PASSWORD'),
                    database=credentials['database']
                )
                cursor = connection.cursor(dictionary=True)
                print("✓ Connected via native MySQL connector")
                return connection, cursor
                
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            raise
    
    def fetch_data(self, datamartid: str, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Fetch data from database or Excel file using datamartid"""
        try:
            # Get credentials
            credentials_list = self.get_credentials(datamartid)
            if not credentials_list:
                raise Exception(f"No credentials found for datamartid: {datamartid}")
            
            # If table_name is specified, filter credentials
            if table_name:
                credentials_list = [cred for cred in credentials_list if cred['table_name'] == table_name]
                if not credentials_list:
                    raise Exception(f"Table {table_name} not found for datamartid: {datamartid}")
            
            # Initialize results
            results = {}
            
            # Process each source
            for credentials in credentials_list:
                try:
                    print(f"\nProcessing source: {credentials['source_type']}")
                    print(f"Table name: {credentials['table_name']}")
                    
                    if credentials['source_type'] == 'Excel':
                        # Handle Excel file
                        file_path = credentials['file_path']
                        print(f"Reading Excel file: {file_path}")
                        
                        if not os.path.exists(file_path):
                            raise Exception(f"Excel file not found: {file_path}")
                            
                        df = pd.read_excel(file_path)
                        print(f"Excel file read successfully. Columns: {df.columns.tolist()}")
                        
                        # Ensure required columns exist
                        if 'date' not in df.columns or 'profit' not in df.columns:
                            raise Exception(f"Excel file must contain 'date' and 'profit' columns. Found columns: {df.columns.tolist()}")
                        # Convert to pandas DataFrame and handle datetime conversion
                        df = pd.DataFrame(data=df, columns=['date', 'profit'])
                        # Explicitly convert date strings to datetime
                        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
                        print(f"DataFrame head:\n{df.head()}\nDataFrame info:\n{df.info()}")
                        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
                        # Ensure date column is datetime type
                        df['date'] = df['date'].dt.date
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        print(f"DataFrame head:\n{df.head()}\nDataFrame info:\n{df.info()}")
                        
                        results[credentials['table_name']] = df.to_dict('records')
                        print("Data processed successfully")
                        
                    else:
                        # Handle SQL table
                        print("Connecting to database...")
                        connection, cursor = self.connect_to_database(credentials)
                        print("Database connection successful")
                        
                        query = f"""
                            SELECT 
                                transaction_date as date,
                                daily_sales as profit
                            FROM {credentials['table_name']}
                            WHERE transaction_date IS NOT NULL
                            ORDER BY transaction_date ASC
                        """
                        print(f"Executing query: {query}")
                        
                        df = pd.read_sql(query, connection)
                        print(f"Query executed successfully. Got {len(df)} rows")
                        
                        # Convert date strings to datetime objects
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        print(f"DataFrame head:\n{df.head()}\nDataFrame info:\n{df.info()}")
                        results[credentials['table_name']] = df.reset_index().to_dict('records')
                        
                        cursor.close()
                        connection.close()
                        print("Database connection closed")
                    
                except Exception as e:
                    print(f"Error processing {credentials['table_name']}: {str(e)}")
                    raise Exception(f"Error processing {credentials['source_type']} source '{credentials['table_name']}': {str(e)}")
            
            return results
            
        except Exception as e:
            print(f"Error in fetch_data: {str(e)}")
            raise