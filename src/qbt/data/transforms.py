import numpy as np 

def log_returns(df, column):
    """Compute clean log returns for a price column."""
    
    # compute raw log returns
    r = np.log(df[column] / df[column].shift(1))
    
    # remove problematic values
    r = r.replace([np.inf, -np.inf], np.nan)   # log(0) or div-by-zero cases
    
    # optional: drop NaNs (usually first row)
    # r = r.dropna()
    
    return r
