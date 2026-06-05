import pandas as pd
import datetime as dt

def create_rfm(df):
    """
    Creates Recency, Frequency, Monetary (RFM) table
    """
    # Reference date = last invoice date + 1 day
    reference_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

    # Create RFM table
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'count',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm


def score_rfm(rfm):
    """
    Assigns R, F, M scores and total RFM score
    """
    quantiles = rfm.quantile(q=[0.25, 0.50, 0.75])

    def rfm_score(x, p):
        if p == 'Recency':
            if x <= quantiles[p][0.25]:
                return 4
            elif x <= quantiles[p][0.50]:
                return 3
            elif x <= quantiles[p][0.75]:
                return 2
            else:
                return 1
        else:
            if x <= quantiles[p][0.25]:
                return 1
            elif x <= quantiles[p][0.50]:
                return 2
            elif x <= quantiles[p][0.75]:
                return 3
            else:
                return 4

    rfm['R'] = rfm['Recency'].apply(rfm_score, args=('Recency',))
    rfm['F'] = rfm['Frequency'].apply(rfm_score, args=('Frequency',))
    rfm['M'] = rfm['Monetary'].apply(rfm_score, args=('Monetary',))

    rfm['RFM_Score'] = rfm[['R', 'F', 'M']].sum(axis=1)

    return rfm
