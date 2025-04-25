from cryptography.fernet import Fernet
import pyodbc
from typing import Dict, Any, Tuple, Optional
import os
from dotenv import load_dotenv

class DatabaseConnection:
    def __init__(self):
        self.key = b'zfis_enruV2dLQxC9OE_ajTb5PP2uqZxahoqncm4RFQ='
        
    def decrypt_password(self, encrypted_password: str) -> str:
        """Decrypt password using stored key"""
        f = Fernet(self.key)
        return f.decrypt(encrypted_password.encode()).decode()
    
    def get_credentials(self, datamartid: str) -> Optional[Dict[str, Any]]:
        """Get database credentials for given datamartid"""
        try:
            # First connect to the insights database to get credentials
            load_dotenv()
            insights_connection = pyodbc.connect(
                f"Driver={{ODBC Driver 17 for SQL Server}};"
                f"Server=logesyssolutions.database.windows.net;"
                f"Database=Insights_DB_Dev;"
                f"UID=lsdbadmin;"
                f"PWD=logesys@1;"
            )
            
            cursor = insights_connection.cursor()
            
            # Query to get credentials
            query = """
                SELECT TableName, FilePath, UserName, Password, Type, SourceType
                FROM m_datamart_tables 
                WHERE datamartid = ?
            """
            
            cursor.execute(query, (datamartid,))
            tables_info = cursor.fetchall()
            
            if not tables_info:
                return None
                
            # Process credentials
            credentials = []
            for table in tables_info:
                table_info = {
                    'table_name': table.TableName,
                    'database': table.FilePath.split('/')[0] if table.FilePath else None,
                    'username': table.UserName,
                    'password': self.decrypt_password(table.Password) if table.Password else None,
                    'source_type': 'Table' if table.Type.lower() == 'table' or table.SourceType.lower() == 'table' else 'Excel'
                }
                credentials.append(table_info)
            
            cursor.close()
            insights_connection.close()
            
            return credentials
            
        except Exception as e:
            raise Exception(f"Error getting credentials: {str(e)}")
    
    def connect_to_database(self, credentials: Dict[str, Any]) -> Tuple[pyodbc.Connection, pyodbc.Cursor]:
        """Connect to database using provided credentials"""
        try:
            connection_string = (
                f"Driver={{ODBC Driver 17 for SQL Server}};"
                f"Server=logesyssolutions.database.windows.net;"
                f"Database={credentials['database']};"
                f"UID={credentials['username']};"
                f"PWD={credentials['password']};"
            )
            
            connection = pyodbc.connect(connection_string)
            cursor = connection.cursor()
            
            return connection, cursor
            
        except Exception as e:
            raise Exception(f"Error connecting to database: {str(e)}")
    
    def fetch_data(self, datamartid: str, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Fetch data from database using datamartid"""
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
            
            # Connect to each database and fetch data
            for credentials in credentials_list:
                connection, cursor = self.connect_to_database(credentials)
                
                # Query to fetch data from table
                query = f"SELECT * FROM {credentials['table_name']}"
                cursor.execute(query)
                
                # Fetch column names
                columns = [column[0] for column in cursor.description]
                
                # Fetch data
                data = cursor.fetchall()
                
                # Convert to dictionary
                table_data = []
                for row in data:
                    table_data.append(dict(zip(columns, row)))
                
                results[credentials['table_name']] = table_data
                
                cursor.close()
                connection.close()
            
            return results
            
        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")