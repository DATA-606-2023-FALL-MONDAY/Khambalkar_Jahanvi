## 1. Title and Author

- **Project Title -** Customer Segmentation using RFM Analysis

- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang

- **Author Name -** Jahanvi Khambalkar

- **GitHub -** https://github.com/Jahanvi-Khambalkar-99

- **LinkedIn -** https://www.linkedin.com/in/jahanvi2299/

- **PowerPoint Presentation file -**

- **Link to YouTube video -** 

    
## 2. Background

- **What is the Project about?**
  - The objective of this project is to implement RFM (Recency, Frequency, Monetary) analysis, a data-driven technique used in marketing and customer analytics to segment a customer base based on their transactional behavior.
  - It involves analyzing three key aspects of customer interactions with a business:
    1. Recency (R): This measures how recently a customer has made a purchase or engaged with your business. It typically involves calculating the time elapsed since the customer's last transaction.
    2. Frequency (F): Frequency indicates how often a customer makes purchases or interacts with your business. It is usually calculated as the total number of transactions within a specified time frame.
    3. Monetary (M): Monetary value represents the total amount of money a customer has spent on your products or services over a given period.

- **Why does it matter?**
    - RFM analysis is a data-driven approach that helps businesses make informed decisions based on actual customer behavior. It minimizes guesswork and intuition, allowing organizations to rely on evidence-based strategie.
    - RFM analysis enables businesses to understand their customers better by categorizing them into distinct segments based on their transactional behavior. This allows for personalized marketing strategies tailored to the specific needs and preferences of each segment.
    - Segment-specific marketing campaigns can be more effective than generic campaigns. RFM analysis helps businesses design and execute campaigns that resonate with each segment, leading to higher response rates and conversion rates.

- **What are your research questions?**
    - What are the distinct customer segments based on their transactional behavior? How can we categorize customers into high-value, low-value, loyal, and at-risk segments?
    - Are there customer segments with growth potential that have been underutilized?
    - What complementary products can be recommended to customers based on their purchase history?
    - Which customer segments have the highest retention rates?
 
      
## 3. Data 

- **About Dataset**

    - The dataset is known as the "Online Retail" from the UCI Machine Learning Repository. It provides transactional information for a UK-based online retail company that specializes in selling unique all-occasion gifts.

- **Data sources -** https://www.kaggle.com/datasets/carrie1/ecommerce-data

- **Data size -** 45.58 MB

- **Data shape -** 541909(rows), 8(columns)

- **Time period -** between 01/12/2010 and 09/12/2011

- **What does each row represent?**
    -  Each row typically represents a single transaction made by a customer with the online retail company.
 
- **Data Dictionary**
  
  Column Name - InvoiceNo
  - Dtype - Categorical (Object)
  - Definition - Code representing each unique transaction. If this code starts with letter 'c', it indicates a cancellation.
  - Potential Values - 25900 unique values
 
  Column Name - StockCode
  - Dtype - Categorical (Object)
  - Definition - Code uniquely assigned to each distinct product.
  - Potential Values - 4070 unique values

  Column Name - Description
  - Dtype - Categorical (Object)
  - Definition - Description of each product.
  - Potential Values - 4223 unique values

  Column Name - Quantity
  - Dtype - Numerical (Integer)
  - Definition - The number of units of a product in a transaction.
  - Potential Values - from negative 80995 to 80995 (the dataset encompasses both positive and negative values. Negative values in this context represent transactions that involve returns or cancellations)

  Column Name - InvoiceDate
  - Dtype - Numerical (Object)
  - Definition - The date and time of the transaction.
  - Potential Values - 23260 unique values

  Column Name - UnitPrice
  - Dtype - Numerical (Float)
  - Definition - The unit price of the product.
  - Potential Values - from negative 11062.06 to 38970 (which suggests the presence of errors or noise in the data, as negative prices don't make sense)

  Column Name - CustomerID
  - Dtype - Categorical (Float)
  - Definition - Identifier uniquely assigned to each customer.
  - Potential Values - IDs range from 12346 to 18287
 
  Column Name - Country
  - Dtype - Categorical (Object)
  - Definition - The country of the customer.
  - Potential Values - 38 unique values.
 
- Which variable/column will be your target/label in your ML model?
    - In this case, there is no traditional target variable, as segmentation is an unsupervised learning task.

- Which variables/columns may be selected as features/predictors for your ML models?
    - The goal is to use unsupervised learning techniques like clustering (e.g., K-means) to group customers based on their transactional behavior, without the need for a specific target variable.
    - Features like Recency (R), Frequency (F), and Monetary (M) will be used as inputs to clustering algorithm to derive the target labels.
