
## Title and Author
- Project Title: Customer Segmentation Based on RFM Analysis
- Prepared for UMBC Data Science Master Degree Capstone under the guidance of Dr. Chaojie (Jay) Wang
- Author Name: Jahanvi Khambalkar
- [GitHub]()
- [LinkedIn]()
- [PowerPoint Presentation]()
- [Youtube Video]()

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

About the data

</details>
 
<details>
<summary><i><b>Data Size and Records</b></i></summary>
  
<br>

RFM analysis is a data-driven approach that helps businesses make informed decisions based on actual customer behavior. It minimizes guesswork and intuition, allowing organizations to rely on evidence-based strategies. RFM analysis enables businesses to understand their customers better by categorizing them into distinct segments based on their transactional behavior. This allows for personalized marketing strategies tailored to the specific needs and preferences of each segment. Segment-specific marketing campaigns can be more effective than generic campaigns. RFM analysis helps businesses design and execute campaigns that resonate with each segment, leading to higher response rates and conversion rates. By identifying and understanding the unique characteristics of customer groups, businesses can optimize resource allocation, enhance customer retention efforts, and ultimately improve the overall effectiveness of their marketing initiatives. This technique is also widely used for customer relationship management.

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
| CustomerID         | Identifier uniquely assigned to each customer                                                                     | object    |
| Country            | The country of the customer                                                                                       | object    |



</details>





