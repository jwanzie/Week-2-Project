### RFM ANALYSIS.
This type of analysis is used to understand and segment customer based on their buying behavior. There are three metrics used to provide info about customer engagement, loyalty and value to the business. These are
- recency; the day a customer made their last order
- frequency; how often the customer made a purchase
- monetary value; the amount spent by the customer in the business. 
To perform RFM, a datset with customer ID, purchase dates and transaction amounts is required. 

The first step taken to begin the analysis is importing the necessary libraries and the dataset

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import datetime as dt

# Import dataset
df = pd.read_csv('rfm_data.csv')
df.head()
```

Next we calculate recency, frequency and monetary value. To calculate recency we have to first convert 'Purchase Date' to datetime. We then calculate recency by subtracting current date from last purchase date. Frequency is calculated by grouping data by 'CustomerID' and counting number of unique 'OrderID' to determine the number of purchase made by each customer. Finally we calculate recency by getting the sum of 'TransactionAmount' in the grouped data to calculate the total amount spent by each customer. 

```python
# Convert 'PurchaseDate' to datetime
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors = 'coerce')
```
```python
# Calculate recency
df_recency = df.groupby(by = 'CustomerID', as_index = False)['PurchaseDate'].max()
df_recency.columns = ['CustomerID', 'LastPurchaseDate']
recent_date = df_recency['LastPurchaseDate'].max()
df_recency['Recency'] = df_recency['LastPurchaseDate'].apply(lambda x: (recent_date - x).days)
df_recency.head()
```
```python
# Calculate frequency
frequency_df = df.drop_duplicates().groupby(by = ['CustomerID'], as_index = False)['OrderID'].count()
frequency_df.columns = ['CustomerID', 'Frequency']
frequency_df.head()
```
```python
# Calculate monetary value
monetary_df = df.groupby(by = 'CustomerID', as_index = False)['TransactionAmount'].sum()
monetary_df.columns = ['CustomerID', 'Monetary']
monetary_df.head()
```

After finding values for recency, frequency and monetary value, we merge all these columns to give one dataframe.

```python
# Merge all dataframe columns
rf_df = df_recency.merge(frequency_df, on = 'CustomerID')
rfm_df = rf_df.merge(monetary_df, on ='CustomerID').drop(columns = 'LastPurchaseDate')
rfm_df.head()
```

Using these values, we can rank customers. We normalize the rank of the customers within the company to analyze their ranking

```python
# Rank curtomers 
rfm_df['R_rank'] = rfm_df['Recency'].rank(ascending = False)
rfm_df['F_rank'] = rfm_df['Frequency'].rank(ascending = True)
rfm_df['M_rank'] = rfm_df['Monetary'].rank(ascending = True)

# Normalize rank
rfm_df['R_rank_norm'] = (rfm_df['R_rank']/rfm_df['R_rank'].max())*100
rfm_df['F_rank_norm'] = (rfm_df['F_rank']/rfm_df['F_rank'].max())*100
rfm_df['M_rank_norm'] = (rfm_df['M_rank']/rfm_df['M_rank'].max())*100

rfm_df.drop(columns = ['R_rank', 'F_rank', 'M_rank'], inplace = True)

rfm_df.head()
```

With this output we can then calculate rfm score for the customers. RFM score is calculated based upon recency, frequency and monetary value normalize ranks. It is according to these scores that customers are segmented on a scale of 5. Formula used for calculating rfm score is : 0.15 * Recency score + 0.28 * Frequency score + 0.57 * Monetary score.

```python
# Calculate rfm score = 0.15*recency score + 0.28*frequency score + 0.57*monetary score
rfm_df['RFM_Score'] = 0.15*rfm_df['R_rank_norm'] + 0.28*rfm_df['F_rank_norm'] + 0.57*rfm_df['M_rank_norm']
rfm_df['RFM_Score'] *= 0.05
rfm_df = rfm_df.round(2)
rfm_df[['CustomerID', 'RFM_Score']].head(7)
```

With the RFM scores generated, we can then rate customers based on the provided scale;
•	rfm score > 4.5 : Top Customer
•	4.5 > rfm score > 4 : High Value Customer
•	4 > rfm score > 3 : Medium value customer
•	3 > rfm score > 1.6 : Low-value customer
•	rfm score < 1.6 : Lost Customer

```python
# Rate customer based on RFM score
rfm_df['Customer_Segment'] = np.where(rfm_df['RFM_Score'] > 4.5, 'Top Customers',
                                     np.where(rfm_df['RFM_Score'] > 4, 'High Value Customer',
                                             np.where(rfm_df['RFM_Score'] > 3, 'Medium Value Customer',
                                                     np.where(rfm_df['RFM_Score'] > 1.6, 'Low Value Customer', 'Lost Customer'))) )
rfm_df[['CustomerID', 'RFM_Score', 'Customer_Segment']].head(20)
```

We can then visualize the customer segments using a pie plot.

```python
# Visualize customer segments
import matplotlib.pyplot as plt

plt.pie(rfm_df.Customer_Segment.value_counts(),
        labels=rfm_df.Customer_Segment.value_counts().index,
        autopct='%.0f%%')
plt.show()
```