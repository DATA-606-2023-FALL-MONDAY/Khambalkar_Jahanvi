
## Title and Author
- Project Title: Customer Segmentation Based on RFM Analysis
- Prepared for UMBC Data Science Master Degree Capstone under the guidance of Dr. Chaojie (Jay) Wang
- Author Name: Jahanvi Khambalkar
- [GitHub]()
- [LinkedIn]()
- [PowerPoint Presentation]()
- [Youtube Video](https://www.youtube.com/watch?v=7ImE07uHMe4)

## Background

<details>
<summary><i><b>What is RFM Analysis?</b></i></summary>
  
<br>

The objective of this project is to implement RFM (Recency, Frequency, Monetary) analysis, a data-driven technique used in marketing and customer analytics to segment a customer base based on their transactional behavior.

It involves analyzing three key aspects of customer interactions with a business:
- Recency (R): This measures how recently a customer has made a purchase or engaged with your business. It typically involves calculating the time elapsed since the customer's last transaction.
- Frequency (F): Frequency indicates how often a customer makes purchases or interacts with your business. It is usually calculated as the total number of transactions within a specified time frame.
- Monetary (M): Monetary value represents the total amount of money a customer has spent on your products or services over a given period.

</details>
 
<details>
<summary><i><b>Why does it matter?</b></i></summary>
  
<br>

RFM analysis is a data-driven approach that helps businesses make informed decisions based on actual customer behavior. It minimizes guesswork and intuition, allowing organizations to rely on evidence-based strategies. RFM analysis enables businesses to understand their customers better by categorizing them into distinct segments based on their transactional behavior. This allows for personalized marketing strategies tailored to the specific needs and preferences of each segment. Segment-specific marketing campaigns can be more effective than generic campaigns. RFM analysis helps businesses design and execute campaigns that resonate with each segment, leading to higher response rates and conversion rates. By identifying and understanding the unique characteristics of customer groups, businesses can optimize resource allocation, enhance customer retention efforts, and ultimately improve the overall effectiveness of their marketing initiatives. This technique is also widely used for customer relationship management.

</details>

<details>
<summary><i><b>Research Questions</b></i></summary>
  
<br>

- What are the distinct customer segments based on their transactional behavior? How can we categorize customers into high-value, low-value, loyal, and at-risk segments?
- Are there customer segments with growth potential that have been underutilized?
- What complementary products can be recommended to customers based on their purchase history?
- Which customer segments have the highest retention rates?

</details>

## Data

<details>
<summary><i><b>Data Source</b></i></summary>
  
<br>

The dataset is known as the [Online Retail](https://doi.org/10.24432/C5BW33) from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/). It provides transactional information for a UK-based online retail company that specializes in selling unique all-occasion gifts.
</details>
 
<details>
<summary><i><b>Data Size and Records</b></i></summary>
  
<br>

- **Data size -** 45.58 MB

- **Data shape -** 541909(rows), 8(columns)

- **Time period -** between 01/12/2010 and 09/12/2011

- Each row typically represents a single transaction made by a customer with the online retail company.

</details>

<details>
<summary><i><b>Data Structure</b></i></summary>
  
<br>

| Column Name                | Definition                                                                                                | Data Type |
|--------------------|-------------------------------------------------------------------------------------------------------------------|-----------|
| InvoiceNo          | Code representing each unique transaction. If this code starts with the letter 'C', it indicates a cancellation   | object    |
| StockCode          | Code uniquely assigned to each distinct product                                                                   | object    |
| Description        | Description of each product                                                                                       | object    |
| Quantity           | The number of units of a product in a transaction                                                                 | integer   |
| InvoiceDate        | The date and time of the transaction                                                                              | object    |
| UnitPrice          | The unit price of the product                                                                                     | float     |
| CustomerID         | Identifier uniquely assigned to each customer                                                                     | float     |
| Country            | The country of the customer                                                                                       | object    |

</details>

## Exploratory Data Analysis & Cleaning

<details>
<summary><i><b>Handling Missing Values</b></i></summary>
  
<br>

<p align="center">
  <img src="https://github.com/DATA-606-2023-FALL-MONDAY/Khambalkar_Jahanvi/blob/main/assets/missing_values.png" alt="missing values" width="auto" height="300">
</p>

<div align="justify">

- The Description and Customer ID together had almost 26% missing values. 
</div>

</details>
 
<details>
<summary><i><b>Handling Duplicate Values</b></i></summary>
  
<br>

<p align="center">
  <img src="https://github.com/DATA-606-2023-FALL-MONDAY/Khambalkar_Jahanvi/blob/main/assets/duplicate_invoice_value.png" alt="duplicate values" width="auto" height="300">
</p>

<div align="justify">

- This chart shows the number of duplicate rows for each invoice no.
- Checked for duplicate values based on the Invoice Number and discovered 5225 duplicate data.

</div>

</details>

<details>
<summary><i><b>Handling Cancelled Transactions</b></i></summary>
  
<br>

<p align="center">
  <img src="https://github.com/DATA-606-2023-FALL-MONDAY/Khambalkar_Jahanvi/blob/main/assets/cancelled_transaction.png" alt="canceled transaction" width="auto" height="300">
</p>

<div align="justify">

- The plot shows that there are around 2% of the transactions in the dataset that has been canceled.

</div>

</details>

<details>
<summary><i><b>Correcting Stockcode Anomalies</b></i></summary>
  
<br>

<p align="center">
  <img src="https://github.com/DATA-606-2023-FALL-MONDAY/Khambalkar_Jahanvi/blob/main/assets/stockcode_occurrence.png" alt="stock code occurrence" width="auto" height="300">
</p>

<div align="justify">
- This plot shows the occurrence of each unique stockcode. 
- Most of the stock codes have 5 or 6 alphanumeric characters. 
</div>

<br>

<p align="center">
  <img src="https://github.com/DATA-606-2023-FALL-MONDAY/Khambalkar_Jahanvi/blob/main/assets/stockcode_anomaly.png" alt="stock code anomaly" width="auto" height="300">
</p>

<div align="justify">
- This data shows that there is 8 stock code anomaly and their occurrence.  
- But then there are stock codes like post, bank charges, and Dot which don't mean anything.  
</div>

</details>

<details>
<summary><i><b>Cleaning Description Column</b></i></summary>
  
<br>

<p align="center">
  <img src="https://github.com/DATA-606-2023-FALL-MONDAY/Khambalkar_Jahanvi/blob/main/assets/description_count.png" alt="description occurrence" width="auto" height="300">
</p>

<div align="justify">
- This plot shows the occurrence of each unique description. 
- Apparently, all the descriptions are in upper case, so that might be the standard form for description. 
- However, on checking the descriptions in lowercase, some descriptions didn't seem normal.
</div>

<br>

<p align="center">
  <img src="https://github.com/DATA-606-2023-FALL-MONDAY/Khambalkar_Jahanvi/blob/main/assets/description_anomaly.png" alt="anomaly descriptions" width="auto" height="300">
</p>

<div align="justify">
- This plot shows that the data have lowercase letters in 19 product descriptions. However, 2 of them seem odd - 'Next Day Carriage' and 'High-Resolution Image'  
</div>

</details>

<details>
<summary><i><b>Treating Zero Unit Price</b></i></summary>
  
<br>

There are 33 rows in the data that have 0.0 as the unit price. 
</details>


## Feature Calculation

<details>
<summary><i><b>Recency</b></i></summary>
  
<br>

The Data is grouped by 'CustomerID', and for each customer, the maximum invoice date is calculated. The recency is then computed as the difference in days between the present date and the maximum invoice date for each customer. 

</details>
 
<details>
<summary><i><b>Frequency</b></i></summary>
  
<br>

It is calculated by removing duplicate rows based on the 'InvoiceNo' column, ensuring that each invoice is counted only once for each customer. Then, it groups the Data by 'CustomerID' and counts the number of unique invoices for each customer.

</details>

<details>
<summary><i><b>Monetary</b></i></summary>
  
<br>

Firstly the 'Total_Amount' is counted by multiplying the 'Quantity' and 'UnitPrice' columns. This gives the total monetary value for each line or transaction in the data frame. The DataFrame is then grouped by 'CustomerID', and for each customer, the total monetary value is calculated by summing the 'Total_Amount' column.

</details>

