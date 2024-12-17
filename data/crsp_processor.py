import pandas as pd
import numpy as np
import wrds
from typing import Optional

class CRSPProcessor:
    """
    Processor for CRSP (Center for Research in Security Prices) data.
    Handles data fetching, cleaning, and preprocessing of stock market data.
    """
    
    def __init__(self, wrds_username: str, file_path: str):
        """
        Initialize the CRSP processor.
        
        Args:
            wrds_username (str): Username for WRDS connection
            file_path (str): Path to save processed data
        """
        self.wrds_username = wrds_username
        self.file_path = file_path
        self.db = None

    def connect_to_wrds(self) -> None:
        """Establish connection to WRDS database."""
        self.db = wrds.Connection(wrds_username=self.wrds_username)

    def fetch_crsp_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch CRSP data from WRDS database.
        
        Args:
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: Raw CRSP data
        """
        query = """
            select 
                a.permno, a.date, a.ret, a.shrout, a.prc, 
                b.siccd, b.exchcd, b.shrcd, 
                c.dlstcd, c.dlret 
            from 
                crsp.msf as a 
                left join crsp.msenames as b 
                    on a.permno=b.permno 
                    and b.namedt<=a.date 
                    and a.date<=b.nameendt 
                left join crsp.msedelist as c 
                    on a.permno=c.permno 
                    and date_trunc('month', a.date) = date_trunc('month', c.dlstdt)
            where 
                date >= '{0}' 
                and date <= '{1}'
        """.format(start_date, end_date)
        
        return self.db.raw_sql(query, date_cols=['date'])

    def process_delisting_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process delisting returns according to standard procedures.
        
        Args:
            df (pd.DataFrame): CRSP dataframe
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        # Set dlret to -0.35 for specific delisting codes
        conditions = [
            (df['dlret'].isnull()) &
            ((df['dlstcd'] == 500) | 
             ((df['dlstcd'] >= 520) & (df['dlstcd'] <= 584))) &
            ((df['exchcd'] == 1) | (df['exchcd'] == 2))
        ]
        df['dlret'] = np.select(conditions, [-0.35], df['dlret'])
        
        # Cap delisting returns at -1
        df['dlret'] = np.where(df['dlret'] < -1, -1, df['dlret'])
        
        # Fill missing delisting returns with 0
        df['dlret'] = df['dlret'].fillna(0)
        
        return df

    def calculate_adjusted_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns adjusted for delisting.
        
        Args:
            df (pd.DataFrame): CRSP dataframe
            
        Returns:
            pd.DataFrame: DataFrame with adjusted returns
        """
        # Calculate return including delisting return
        df['retadj'] = (1 + df['ret']) * (1 + df['dlret']) - 1
        
        # Use dlret when return is missing but dlret is not 0
        df['retadj'] = np.where(
            (df['ret'].isnull()) & (df['dlret'] != 0),
            df['dlret'],
            df['retadj']
        )
        
        # Convert to percentage
        df['retadj'] = df['retadj'] * 100
        
        return df

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main data processing pipeline.
        
        Args:
            df (pd.DataFrame): Raw CRSP data
            
        Returns:
            pd.DataFrame: Processed data ready for analysis
        """
        # Process delisting returns
        df = self.process_delisting_returns(df)
        
        # Calculate adjusted returns
        df = self.calculate_adjusted_returns(df)
        
        # Calculate market equity
        df['me'] = df['prc'].abs() * df['shrout'] / 1000
        
        # Convert date to yyyymm format
        df['yyyymm'] = df['date'].dt.year * 100 + df['date'].dt.month
        
        # Filter data
        df = df[
            (df['me'] != 0) & 
            df['me'].notna() & 
            df['retadj'].notna()
        ]
        
        # Select and convert columns
        df = df[[
            'permno', 'yyyymm', 'retadj', 'me', 
            'shrcd', 'prc', 'siccd', 'exchcd'
        ]]
        
        # Convert data types
        df['permno'] = df['permno'].astype(int)
        df['yyyymm'] = df['yyyymm'].astype(int)
        
        return df

    def merge_with_ff_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge CRSP data with Fama-French factors.
        
        Args:
            df (pd.DataFrame): Processed CRSP data
            
        Returns:
            pd.DataFrame: Data merged with FF factors
        """
        # Load Fama-French factors
        ff = pd.read_csv(self.file_path + 'ff.csv', index_col=0)
        rf_dict = ff['RF'].to_dict()
        
        # Merge and calculate excess returns
        df['rf'] = df['yyyymm'].map(rf_dict)
        df['exret'] = df['retadj'] - df['rf']
        
        return df

    def run_full_process(self, start_date: str, end_date: str) -> None:
        """
        Run the full CRSP data processing pipeline.
        
        Args:
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
        """
        # Connect to WRDS if not already connected
        if self.db is None:
            self.connect_to_wrds()
        
        # Fetch raw data
        crsp_raw = self.fetch_crsp_data(start_date, end_date)
        
        # Process data
        crsp_processed = self.process_data(crsp_raw)
        
        # Merge with FF factors
        crsp_final = self.merge_with_ff_factors(crsp_processed)
        
        # Save to file
        crsp_final.to_csv(self.file_path + 'crsp.csv', index=False)