

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
import pickle
import streamlit as st
import os
from datetime import datetime
import squarify
import base64

# # Loading the saved model

# load_model = pickle.load(open('C:/Users/Jahanvi/OneDrive/Documents/Studies/606 - Capstone Project/customer_segmentation/trained_model_clustering.sav', 'rb'))


# GUI setup
st.title("Capstone Project")
st.header("Customer Segmentation based on RFM Analysis", divider='rainbow')

menu = ["Business Understanding", "Data Understanding","Data Cleaning","Calculating Features", "Modeling", "Performance Evaluation", "Conclusion and Limitation"] # , "BigData: Spark"
choice = st.sidebar.selectbox('Menu', menu)

def load_data(uploaded_file):
    if uploaded_file is not None:
        st.sidebar.success("File uploaded successfully!")
        df = pd.read_csv(uploaded_file, encoding='latin-1', sep='\s+', header=None, names=['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice',	'CustomerID', 'Country'])
        df.to_csv("CDNOW_master_new.txt", index=False)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%Y%m%d')
        st.session_state['df'] = df
        return df
    else:
        st.write("Please upload a data file to proceed.")
        return None

def csv_download_link(df, csv_file_name, download_link_text):
    csv_data = df.to_csv(index=True)
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{csv_file_name}">{download_link_text}</a>'
    st.markdown(href, unsafe_allow_html=True)    
# Initializing session state variables
if 'df' not in st.session_state:
    st.session_state['df'] = None

if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None


# Main Menu
if choice == 'Business Understanding':
    st.subheader("Business Objective")
    st.write("""
    ###### Customer segmentation is a fundamental task in marketing and customer relationship management. With the advancements in data analytics and machine learning, it is now possible to group customers into distinct segments with a high degree of precision, allowing businesses to tailor their marketing strategies and offerings to each segment's unique needs and preferences.

    ###### Through this customer segmentation, businesses can achieve:
    - **Personalization**: Tailoring marketing strategies to meet the unique needs of each segment.
    - **Optimization**: Efficient allocation of marketing resources.
    - **Insight**: Gaining a deeper understanding of the customer base.
    - **Engagement**: Enhancing customer engagement and satisfaction.
    """)    
    st.subheader("Problem Statement")      
    st.write("""
    ###### Utilize data analysis and machine learning techniques in Python to perform customer segmentation based on RFM Analysis.
    """)
    
    st.subheader("Key Features")      
    st.write("""
    - **RECENCY** - Measures how recently a customer has made a purchase or engaged with the business. Calculate the time elapsed since the customer's last transaction.
    - **FREQUENCY** - Indicates how often a customer makes purchases or interacts with the business. Calculate the total number of transactions within a specified time frame.
    - **MONETARY** - Represent the total amount of money a customer has spent on products or services. Sum the monetary value of all transactions within a given period.

    """)
    # st.image("Customer-Segmentation.png", caption="Customer Segmentation", use_column_width=True)

elif choice == 'Data Understanding':    
    st.write("### Sample Data Can be found here - ")
    # Liệt kê tất cả các file trong thư mục 'sample_data'
    sample_files = os.listdir('dataset')
    
    # Tạo một radio button để cho phép người dùng chọn giữa việc sử dụng file mẫu hoặc tải lên file mới
    data_source = st.sidebar.radio('Data source', ['Use a sample file'])
    
    if data_source == 'Use a sample file':
        st.session_state['uploaded_file'] = st.sidebar.file_uploader("Choose a file", type=['txt'])
        
        if st.session_state['uploaded_file'] is not None:
            load_data(st.session_state['uploaded_file'])
        # Cho phép người dùng tải lên một file mới
        

    # st.session_state['uploaded_file'] = st.sidebar.file_uploader("Choose a file", type=['txt'])
    # load_data(st.session_state['uploaded_file'])
    
    if st.session_state['df'] is not None:
        st.write("### Data Overview")
        st.write("Number of rows:", st.session_state['df'].shape[0])
        st.write("Number of columns:", st.session_state['df'].shape[1])
        st.write("First five rows of the data:")
        st.write(st.session_state['df'].head())

elif choice == 'Data Cleaning': 
    
    st.title("Data Cleaning")
    if st.session_state['df'] is not None:
        # 1. Handling missing
        st.write("### 1. Handling Missing Values")
        st.write("Number of missing values")
        st.write(st.session_state['df'].isnull().sum())
        
        

        # Streamlit app
        st.write('Missing Values Analysis')
        missing_values = st.session_state['df'].isnull().sum()

        # Barh chart
        fig_percentage_of_missing_value_in_each_column, ax = plt.subplots(figsize=(15, 4))
        ax.barh(missing_values.index, ((missing_values / st.session_state['df'].shape[0]) * 100), color='#ff6200')
        plt.xlabel('Percentage')
        plt.ylabel('Columns')
        plt.title('Percentage of missing values in each column')

        # Display the chart using Streamlit
        st.pyplot(fig_percentage_of_missing_value_in_each_column)

        # Providing options for handling missing 
        if st.checkbox('Remove rows with missing values'):
            st.session_state['df'] = st.session_state['df'].dropna()
            st.write("Rows with missing values removed.")
            # Display the updated DataFrame shape
            st.subheader('Updated DataFrame Shape')
            st.write(st.session_state['df'].shape)
          
            
        
        
        st.write("### 2. Handling Null Values")
        # 2. Handling Null Values
        st.write("Number of NA values")
        st.write((st.session_state['df'] == 'NA').sum())
        if st.checkbox('Remove rows with NA values'):
            st.session_state['df'].replace('NA', pd.NA, inplace=True)
            st.session_state['df'].dropna(inplace=True)
            st.write("Rows with NA values removed.")
            st.subheader('Updated DataFrame Shape')
            st.write(st.session_state['df'].shape)

        
        st.write("### 3. Handling Duplicate Values")

        # Calculate missing values
        # 3. Handleing Duplicate values
        st.write("Number of duplicate rows:", st.session_state['df'].duplicated().sum())
        
        # Calculate and display the number of duplicated values for each invoice number
        duplicated_values = st.session_state['df'][st.session_state['df'].duplicated(subset=['InvoiceNo', 'CustomerID'], keep=False)].groupby('InvoiceNo')['CustomerID'].count().reset_index()
        duplicated_values.columns = ['InvoiceNo', 'Duplicated Values Count']

        st.write("Duplicated Values Count for Each Invoice Number (Other than Original):")
        st.dataframe(duplicated_values)
        
        # Providing options for handling missing and duplicate values
        if st.checkbox('Remove duplicate rows'):
            st.session_state['df'].drop_duplicates(inplace=True)
            st.write("Duplicate rows removed.")
            st.subheader('Updated DataFrame Shape')
            st.write(st.session_state['df'].shape)

        # 4. Handeling the cancelled transaction
        st.write("### 4. Handling the cancelled transaction")
        # Lets first see the number of cancelled transactions
        st.session_state['df']['InvoiceNo'] = st.session_state['df']['InvoiceNo'].astype(str)
        # Display the number of cancelled transactions
        cancelled_count = st.session_state['df'][st.session_state['df']['InvoiceNo'].str.startswith('C')]['InvoiceNo'].count()
        st.write(f'The number of cancelled transactions is: {cancelled_count}')

     
        # plot for percentage of cancelled transaction
        # Calculate the total number of transactions
        total_transactions = len(st.session_state['df'])

        # Calculate the number of canceled transactions (those starting with 'C')
        canceled_transactions = len(st.session_state['df'][st.session_state['df']['InvoiceNo'].str.startswith('C')])

        # Calculate the percentage of canceled transactions
        cancel_percentage = (canceled_transactions / total_transactions) * 100

        # Create a bar plot
        fig_percentage_of_cancelled_transaction, ax = plt.subplots()
        ax.bar(['Canceled', 'Successful'], [cancel_percentage, 100 - cancel_percentage])
        ax.set_xlabel('Transaction Type')
        ax.set_ylabel('Percentage')
        ax.set_title('Percentage of Canceled vs. Successful Transactions')

        # Add percentage markers on top of the bars
        for x, y in zip(['Canceled', 'Successful'], [cancel_percentage, 100 - cancel_percentage]):
            ax.text(x, y, f'{y:.2f}%', ha='center', va='bottom')

        # Display the plot using Streamlit
        st.pyplot(fig_percentage_of_cancelled_transaction)
        
        # Providing options for handling missing and duplicate values
        if st.checkbox('Remove Cancelled Transactions'):
            st.session_state['df'] = st.session_state['df'][~st.session_state['df']['InvoiceNo'].str.startswith('C')]
            st.write("Cancelled Transactions removed.")
            st.subheader('Updated DataFrame Shape')
            st.write(st.session_state['df'].shape)
        
        
        

        # 5. Handelling Stock code anomlies
        st.write("### 5. Handling Stock code anomlies")
        # Calculate and display the number of unique stock codes
        unique_stockcodes = pd.DataFrame(st.session_state['df']['StockCode'].unique(), columns=['Unique Stock Codes'])
        st.write("Number of Unique Stock Codes:", len(unique_stockcodes))
        # st.write()
        
        #Finding the occurrence of each unique stock code
        st.write("Occurrence of each unique stock code:")
         # Calculate and display the occurrence of each unique stock code
        stock_code_counts = st.session_state['df']['StockCode'].value_counts().reset_index()
        stock_code_counts.columns = ['Stock Code', 'Occurrence']
        st.dataframe(stock_code_counts)

        # Identify and filter anomalous stock codes
        anomalous_stockcodes_list = ['POST', 'C2', 'M', 'BANK CHARGES', 'PADS', 'DOT']
        anomalous_df = st.session_state['df'][st.session_state['df']['StockCode'].isin(anomalous_stockcodes_list)]

        # Display anomalous stock codes
        st.write('Anomalous Stock Codes:')
        st.write(anomalous_df)

        # Remove anomalous stock codes
        # filtered_df = st.session_state['df'][~st.session_state['df']['StockCode'].isin(anomalous_stockcodes_list)]

        # st.write("anorm stockcode removed:")

        # Providing options for handling missing and duplicate values
        if st.checkbox('Remove Rows with Stockcode Anomalies'):
            st.session_state['df'] = st.session_state['df'][~st.session_state['df']['StockCode'].isin(anomalous_stockcodes_list)]
            st.write("Rows with Stockcode Anomalies removed.")
            st.subheader('Updated DataFrame Shape')
            st.write(st.session_state['df'].shape)
        








        st.write("### 6. Cleaning Description Column")

        # 5. handelling bad discription Streamlit app
        st.write('Description Value Counts')
     
        value_counts = st.session_state['df']['Description'].value_counts()
        st.write(value_counts)

        # Bar chart for visualization
        st.bar_chart(value_counts)
        
        # Identify and filter anomalous discription
        anomalous_description_list = ['NextDayCarriage', 'HighResolutionImage']
        anomalous_description_df = st.session_state['df'][st.session_state['df']['Description'].isin(anomalous_description_list)]
        
        # Display anomalous description
        st.write('Anomalous Description:')
        st.write(anomalous_description_df)
        
        # Providing options for handling missing and duplicate values
        if st.checkbox('Remove the columns with non-normal description values'):
            st.session_state['df'] = st.session_state['df'][~st.session_state['df']['Description'].isin(['Next Day Carriage', 'High Resolution Image'])]
            st.write("columns with non-normal description values removed.")
            st.subheader('Updated DataFrame Shape')
            st.write(st.session_state['df'].shape)
            
            
            
        st.write("### 7. Treating Zero at Unit Price")

        # 7. Handelling 0 as unit price
        # Calculate value counts of unique unit prices
        unit_price_counts = st.session_state['df']['UnitPrice'].value_counts()

        # Create a DataFrame with counts as a column
        unique_unitprice = pd.DataFrame({'UnitPrice': unit_price_counts.index, 'Count': unit_price_counts.values})
        unique_unitprice = unique_unitprice.sort_values(by='UnitPrice')

        # Streamlit app
        st.write('Unique Unit Price Counts')

        # Display the DataFrame
        st.dataframe(unique_unitprice, width=800, height=600)
        
        # Providing options for handling missing and duplicate values
        if st.checkbox('Remove the rows with Unit price = 0'):
            st.session_state['df'] = st.session_state['df'][st.session_state['df']['UnitPrice'] != 0]
            st.write("Rows with Unit price = 0 removed.")
            st.subheader('Updated DataFrame Shape')
            st.write(st.session_state['df'].shape)
        
        
        
        
        
        # st.title("Unique Values After Cleaning")


        # # 2. Display number of unique values for each column
        # st.write("Number of unique values for each column:")
        # st.write(st.session_state['df'].nunique())

elif choice == 'Calculating Features': 
    
        st.write("Cleaned Dataframe ")
        st.write(st.session_state['df'])
        st.write(st.session_state['df'].shape)
        
        
        # Streamlit App
        st.title('Recency Calculator')
        # Button to calculate Recency 
        if st.checkbox('Recency'):
            present_date = '2012-01-01'
            present_date = pd.to_datetime(present_date)
            # st.session_state['df']['InvoiceDate'] = pd.to_datetime(st.session_state['df']['InvoiceDate'])

            # Calculating Recency
            recency = st.session_state['df'].groupby(['CustomerID']).agg({'InvoiceDate': lambda x: ((present_date - x.max()).days)})

            # Renaming the Invoice date column to Recency
            recency.rename(columns={"InvoiceDate": "Recency"}, inplace=True)

            

            # Display the calculated Recency
            st.subheader('Recency DataFrame')
            st.write(recency)
            recency_df = pd.DataFrame(recency)
            st.session_state['recency_df'] = recency_df
            
        import numpy as np    
        st.title('Frequency Calculator')
        # Button to calculate Recency 
        if st.checkbox('Frequency'):
            # Calculate Frequency
            frequency = st.session_state['df'].groupby(['CustomerID'])[['InvoiceNo']].count()
            # Set a seed for reproducibility
            # Customer IDs
            # customer_ids = ["13042", "13043", "13044", "13045", "13046", "13047", "13048", "13049", "17850"]
            # freq_values = [2, 59, 3, 6, 20, 1, 30, 4, 7]
            # frequency = np.random.randint(1, 51, size=9)
            # Renaming the column
            frequency.rename(columns={'InvoiceNo': 'Frequency'}, inplace=True)

            # Displaying Frequency DataFrame
            st.subheader('Frequency DataFrame')
            st.write(frequency)
            frequency_df = pd.DataFrame(frequency)

            # frequency_df = pd.DataFrame({'CustomerID': customer_ids,'Frequency' :freq_values}, index=customer_ids)
            st.session_state['frequency_df'] = frequency_df
            # st.write(frequency_df)
            
            
            
        st.title('Monetary Calculator')
        # Button to calculate Recency 
        if st.checkbox('Monetary'):
            # Calculate Monetary
            # Calculate Total Amount
            st.session_state['df']["Total_Amount"] = st.session_state['df']["Quantity"] * st.session_state['df']["UnitPrice"]
            monetary = st.session_state['df'].groupby(["CustomerID"])[["Total_Amount"]].sum()

            # Renaming the column
            monetary.rename(columns={'Total_Amount': 'Monetary'}, inplace=True)

            # Displaying Monetary DataFrame
            st.subheader('Monetary DataFrame')
            st.write(monetary)    
            monetary_df = pd.DataFrame(monetary)
            st.session_state['monetary_df'] = monetary_df
            
elif choice == 'Modeling': 
        

        df_rfm = pd.concat([st.session_state['recency_df'], st.session_state['frequency_df'], st.session_state['monetary_df']], axis=1)
        st.session_state['df_rfm'] = df_rfm
        
        
        # random_numbers = np.random.randint(1, 51, size=7)

        st.session_state['df_rfm'] = st.session_state['df_rfm']
        st.subheader('Merged Dataframe')
        
        st.write(st.session_state['df_rfm'])
        


        # Calculate quantiles
        quantiles = st.session_state['df_rfm'].quantile(q=[0.25, 0.5, 0.75])

        # Scoring functions
        def r_score(r):
            if r <= quantiles['Recency'][0.25]:
                return 4
            elif r <= quantiles['Recency'][0.50]:
                return 3
            elif r <= quantiles['Recency'][0.75]:
                return 2
            else:
                return 1

        def fm_score(f, m, quantile):
            if f <= quantile[0.25]:
                return 1
            elif f <= quantile[0.50]:
                return 2
            elif f <= quantile[0.75]:
                return 3
            else:
                return 4

        # Apply scoring functions to create RFM segments
        st.session_state['df_rfm']['R_Score'] = st.session_state['df_rfm']['Recency'].apply(r_score)
        st.session_state['df_rfm']['F_Score'] = st.session_state['df_rfm']['Frequency'].apply(fm_score, args=('Frequency', quantiles['Frequency']))
        st.session_state['df_rfm']['M_Score'] = st.session_state['df_rfm']['Monetary'].apply(fm_score, args=('Monetary', quantiles['Monetary']))

        st.subheader('RFM DataFrame with Scores')
        st.write(st.session_state['df_rfm'])
        
        
        st.header('Method 1 : Calculating the Combined RFM Score & Segmenting based on combined RFM score.')
        
        # for that creating a copy of df_rfm
        st.session_state['df_rfm_score'] = st.session_state['df_rfm'].copy()
        # Calculating RFM Score
        st.session_state['df_rfm_score']['RFM_Score'] = st.session_state['df_rfm_score']['R_Score'] + st.session_state['df_rfm_score']['F_Score'] + st.session_state['df_rfm_score']['M_Score']
        st.write(st.session_state['df_rfm_score'])
        
        
        st.subheader('Summary Statistics RFM DataFrame with Scores')
        st.write(st.session_state['df_rfm_score'].describe())
        
        st.subheader('RFM DataFrame with Scores with Value Segments')

        # Create RFM segments based on the RFM score
        segment_labels = ['Low-Value', 'Mid-Value', 'High-Value']
        st.session_state['df_rfm_score']['Value_Segment'] = pd.qcut(st.session_state['df_rfm_score']['RFM_Score'], q=3, labels=segment_labels)
        st.write(st.session_state['df_rfm_score'])
        
        st.subheader('RFM Value Segment Distribution')

        # RFM Segment Distribution
        segment_counts = st.session_state['df_rfm_score']['Value_Segment'].value_counts().reset_index()
        segment_counts.columns = ['Value_Segment', 'Count']

        pastel_colors = px.colors.qualitative.Pastel

        # Create the bar chart
        fig_segment_dist = px.bar(segment_counts, x='Value_Segment', y='Count', 
                                color='Value_Segment', color_discrete_sequence=pastel_colors,
                                title='', width=400, height=400)

        # Update the layout
        fig_segment_dist.update_layout(xaxis_title='RFM Value Segment',
                                    yaxis_title='Count',
                                    showlegend=False)
        st.plotly_chart(fig_segment_dist)

        st.subheader('RFM DataFrame with Scores with Customer Segment')

        # Create a new column for RFM Customer Segments
        st.session_state['df_rfm_score']['RFM_Customer_Segments'] = ''

        # Assign RFM segments based on the RFM score
        st.session_state['df_rfm_score'].loc[st.session_state['df_rfm_score']['RFM_Score'] >= 11, 'RFM_Customer_Segments'] = 'Champions'
        st.session_state['df_rfm_score'].loc[(st.session_state['df_rfm_score']['RFM_Score'] >= 10) & (st.session_state['df_rfm_score']['RFM_Score'] < 11), 'RFM_Customer_Segments'] = 'Loyal'
        st.session_state['df_rfm_score'].loc[(st.session_state['df_rfm_score']['RFM_Score'] >= 8) & (st.session_state['df_rfm_score']['RFM_Score'] < 10), 'RFM_Customer_Segments'] = 'Potential Loyalists'
        st.session_state['df_rfm_score'].loc[(st.session_state['df_rfm_score']['RFM_Score'] >= 6) & (st.session_state['df_rfm_score']['RFM_Score'] < 8), 'RFM_Customer_Segments'] = 'Small Buyer/Cannot Lose'
        st.session_state['df_rfm_score'].loc[(st.session_state['df_rfm_score']['RFM_Score'] >= 5) & (st.session_state['df_rfm_score']['RFM_Score'] < 6), 'RFM_Customer_Segments'] = 'At Risk'
        st.session_state['df_rfm_score'].loc[(st.session_state['df_rfm_score']['RFM_Score'] >= 4) & (st.session_state['df_rfm_score']['RFM_Score'] < 5), 'RFM_Customer_Segments'] = "Need Attention"
        st.session_state['df_rfm_score'].loc[(st.session_state['df_rfm_score']['RFM_Score'] >= 3) & (st.session_state['df_rfm_score']['RFM_Score'] < 4), 'RFM_Customer_Segments'] = "Lost"

        st.write(st.session_state['df_rfm_score'])


        st.subheader('Customer Segment Counts')
        st.write(st.session_state['df_rfm_score']['RFM_Customer_Segments'].value_counts())
        
        
        st.subheader('Customer Segments based on RFM Score')
        
        segment_product_counts = st.session_state['df_rfm_score'].groupby(['Value_Segment', 'RFM_Customer_Segments']).size().reset_index(name='Count')

        segment_product_counts = segment_product_counts.sort_values('Count', ascending=False)

        fig_treemap_segment_product = px.treemap(segment_product_counts, 
                                                path=['Value_Segment', 'RFM_Customer_Segments'], 
                                                values='Count',
                                                color='Value_Segment', color_discrete_sequence=px.colors.qualitative.Pastel,
                                                title='')
        st.plotly_chart(fig_treemap_segment_product)
        # fig_treemap_segment_product.show()

        st.subheader('Comparison of RFM Segments')

        import plotly.colors
        import plotly.graph_objects as go

        pastel_colors = plotly.colors.qualitative.Pastel

        segment_counts = st.session_state['df_rfm_score']['RFM_Customer_Segments'].value_counts()

        # Create a bar chart to compare segment counts
        fig_compare_segment_counts = go.Figure(data=[go.Bar(x=segment_counts.index, y=segment_counts.values,
                                    marker=dict(color=pastel_colors))])

        # Set the color of the Champions segment as a different color
        champions_color = 'rgb(158, 202, 225)'
        fig_compare_segment_counts.update_traces(marker_color=[champions_color if segment == 'Champions' else pastel_colors[i]
                                        for i, segment in enumerate(segment_counts.index)],
                        marker_line_color='rgb(8, 48, 107)',
                        marker_line_width=1.5, opacity=0.6)

        # Update the layout
        fig_compare_segment_counts.update_layout(title='',
                        xaxis_title='RFM Segments',
                        yaxis_title='Number of Customers',
                        showlegend=False)

        st.plotly_chart(fig_compare_segment_counts)

        st.header('Method 2 : Using individual Recency, Frequency and Monetary features')

        st.session_state['df_cluster'] = st.session_state['df_rfm'].copy()
        
        st.subheader('Scaling the data')

        from sklearn.preprocessing import StandardScaler

        # Initialize the StandardScaler
        scaler = StandardScaler()

        # List of columns that don't need to be scaled
        columns_to_exclude = ['R_Score', 'F_Score', 'M_Score']

        # List of columns that need to be scaled
        columns_to_scale = st.session_state['df_cluster'].columns.difference(columns_to_exclude)

        # Copy the cleaned dataset
        st.session_state['df_cluster_scaled'] = st.session_state['df_cluster'].copy()

        # Applying the scaler to the necessary columns in the dataset
        st.session_state['df_cluster_scaled'][columns_to_scale] = scaler.fit_transform(st.session_state['df_cluster_scaled'][columns_to_scale])

        # Display the first few rows of the scaled data
        st.write(st.session_state['df_cluster_scaled'])
        
        st.subheader('Clustering the unlabled training data using K-Mean')

        
        from sklearn.cluster import KMeans
        from matplotlib import pyplot as plt
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=3)
        st.session_state['df_cluster_scaled']['Cluster'] = kmeans.fit_predict(st.session_state['df_cluster_scaled'][['Recency','Frequency','Monetary', 'R_Score', 'F_Score', 'M_Score']])
        st.write(st.session_state['df_cluster_scaled'])

        st.subheader('3D Visualization of Customer Clusters in RFM Features')

        
        # 3D Scattered visualization of points in each cluster
        import plotly.graph_objects as go

        # Setting up the color scheme for the clusters (RGB order)
        colors = ['#e8000b', '#1ac938', '#023eff', '#00ffff', '#07800f']

        # Create separate data frames for each cluster
        cluster_low = st.session_state['df_cluster_scaled'][st.session_state['df_cluster_scaled']['Cluster'] == 0]
        cluster_mid = st.session_state['df_cluster_scaled'][st.session_state['df_cluster_scaled']['Cluster'] == 1]
        cluster_high = st.session_state['df_cluster_scaled'][st.session_state['df_cluster_scaled']['Cluster'] == 2]


        # Create a 3D scatter plot
        fig_3d = go.Figure()

        # Add data points for each cluster separately and specify the color
        fig_3d.add_trace(go.Scatter3d(x=cluster_low['Recency'], y=cluster_low['Frequency'], z=cluster_low['Monetary'], 
                                mode='markers', marker=dict(color=colors[0], size=5, opacity=0.4), name='Low Value'))
        fig_3d.add_trace(go.Scatter3d(x=cluster_mid['Recency'], y=cluster_mid['Frequency'], z=cluster_mid['Monetary'], 
                                mode='markers', marker=dict(color=colors[1], size=5, opacity=0.4), name='Mid Value'))
        fig_3d.add_trace(go.Scatter3d(x=cluster_high['Recency'], y=cluster_high['Frequency'], z=cluster_high['Monetary'], 
                                mode='markers', marker=dict(color=colors[2], size=5, opacity=0.4), name='High Value'))
        # fig.add_trace(go.Scatter3d(x=cluster_high_max['Recency'], y=cluster_high_max['Frequency'], z=cluster_high_max['Monetary'], 
        #                            mode='markers', marker=dict(color=colors[3], size=5, opacity=0.4), name='High Value Max'))
        # fig.add_trace(go.Scatter3d(x=cluster_high_pro['Recency'], y=cluster_high_pro['Frequency'], z=cluster_high_pro['Monetary'], 
        #                            mode='markers', marker=dict(color=colors[4], size=5, opacity=0.4), name='High Value'))
        # Set the title and layout details
        fig_3d.update_layout(
            title=dict(text='3D Visualization of Customer Clusters in RFM Features', x=0.5),
            scene=dict(
                xaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='Recency'),
                yaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='Frequency'),
                zaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='Monetary'),
            ),
            width=900,
            height=800
        )

        # Show the plot
        st.plotly_chart(fig_3d)
        
        st.subheader('3D Visualization of Customer Clusters in RFM Features')

        
        # creating separate dataframe of each cluster
        st.session_state['df_Custer_Low_Value'] = st.session_state['df_cluster_scaled'].loc[st.session_state['df_cluster_scaled']['Cluster'] == 0]
        st.session_state['df_Custer_Mid_Value'] = st.session_state['df_cluster_scaled'].loc[st.session_state['df_cluster_scaled']['Cluster'] == 1]
        st.session_state['df_Custer_High_Value'] = st.session_state['df_cluster_scaled'].loc[st.session_state['df_cluster_scaled']['Cluster'] == 2]
                
        # Defining Min Max Threshold for Cluster 0 - Low Value
        describe_25_for_clusterLow_0_recency  = st.session_state['df_Custer_Low_Value'].describe().iloc[4].Recency
        describe_25_for_clusterLow_0_frequency = st.session_state['df_Custer_Low_Value'].describe().iloc[4].Frequency
        describe_25_for_clusterLow_0_monetary = st.session_state['df_Custer_Low_Value'].describe().iloc[4].Monetary
        describe_75_for_clusterLow_0_recency  = st.session_state['df_Custer_Low_Value'].describe().iloc[6].Recency
        describe_75_for_clusterLow_0_frequency = st.session_state['df_Custer_Low_Value'].describe().iloc[6].Frequency
        describe_75_for_clusterLow_0_monetary = st.session_state['df_Custer_Low_Value'].describe().iloc[6].Monetary

        # Defining Min Max Threshold for Cluster 1 - Mid Value
        describe_25_for_clusterMid_1_recency  = st.session_state['df_Custer_Mid_Value'].describe().iloc[4].Recency
        describe_25_for_clusterMid_1_frequency = st.session_state['df_Custer_Mid_Value'].describe().iloc[4].Frequency
        describe_25_for_clusterMid_1_monetary = st.session_state['df_Custer_Mid_Value'].describe().iloc[4].Monetary
        describe_75_for_clusterMid_1_recency  = st.session_state['df_Custer_Mid_Value'].describe().iloc[6].Recency
        describe_75_for_clusterMid_1_frequency = st.session_state['df_Custer_Mid_Value'].describe().iloc[6].Frequency
        describe_75_for_clusterMid_1_monetary = st.session_state['df_Custer_Mid_Value'].describe().iloc[6].Monetary
        
        # Segmentation of Low Value Cluster
        st.session_state['df_Custer_Low_Value'].loc[(st.session_state['df_Custer_Low_Value']['Recency'] >= describe_75_for_clusterLow_0_recency) & (st.session_state['df_Custer_Low_Value']['Frequency'] <= describe_25_for_clusterLow_0_frequency) & (st.session_state['df_Custer_Low_Value']['Monetary'] <= describe_25_for_clusterLow_0_monetary), 'Customer_Segments'] = 'Lost'
        st.session_state['df_Custer_Low_Value'].loc[(st.session_state['df_Custer_Low_Value']['Recency'] <= describe_25_for_clusterLow_0_recency) & (st.session_state['df_Custer_Low_Value']['Frequency'] >= describe_75_for_clusterLow_0_frequency) & (st.session_state['df_Custer_Low_Value']['Monetary'] >= describe_75_for_clusterLow_0_frequency), 'Customer_Segments'] = 'Need Attention'
        # Filling the value for 3rd segment
        st.session_state['df_Custer_Low_Value']['Customer_Segments'].fillna('At Risk', inplace = True)
                
                
        # Segmentation of Mid Value Cluster
        st.session_state['df_Custer_Mid_Value'].loc[(st.session_state['df_Custer_Mid_Value']['Recency'] >= describe_75_for_clusterMid_1_recency) & (st.session_state['df_Custer_Mid_Value']['Frequency'] <= describe_25_for_clusterMid_1_frequency) & (st.session_state['df_Custer_Mid_Value']['Monetary'] <= describe_25_for_clusterMid_1_monetary), 'Customer_Segments'] = 'Small Buyers/Cannot Lose'
        st.session_state['df_Custer_Mid_Value'].loc[(st.session_state['df_Custer_Mid_Value']['Recency'] <= describe_25_for_clusterMid_1_recency) & (st.session_state['df_Custer_Mid_Value']['Frequency'] >= describe_75_for_clusterMid_1_frequency) & (st.session_state['df_Custer_Mid_Value']['Monetary'] >= describe_75_for_clusterMid_1_monetary), 'Customer_Segments'] = 'Loyal'
        # Filling the value for 3rd segment
        st.session_state['df_Custer_Mid_Value']['Customer_Segments'].fillna('Potential Loyalist', inplace = True)

        # In High Value Cluster there can be only on segment 
        st.session_state['df_Custer_High_Value']['Customer_Segments'] = 'Champions'

        st.subheader('Segmentation within the cluster based on the feature variations')

        # merging the dataframe separate Dataframe of 3 clusters 
        st.session_state['df_cluster_scaled_segment'] = pd.concat([st.session_state['df_Custer_Low_Value'], st.session_state['df_Custer_Mid_Value'], st.session_state['df_Custer_High_Value']])
        st.session_state['df_cluster_scaled_segment']


        st.subheader('Customer Segments based on RFM Features')


        # Tree map of Customer Segments
        segment_customer_counts_tree = st.session_state['df_cluster_scaled_segment'].groupby(['Cluster', 'Customer_Segments']).size().reset_index(name='Count')
        pastel_colors = px.colors.qualitative.Pastel

        segment_customer_counts_tree = segment_customer_counts_tree.sort_values('Count', ascending=False)

        fig_treemap_segment_customers = px.treemap(segment_customer_counts_tree, 
                                                path=['Cluster', 'Customer_Segments'], 
                                                values='Count',
                                                color='Customer_Segments', color_discrete_map={'(?)':'lightgray', 'Potential Loyalist':'plum', 'Loyal':'thistle', 'Small Buyers/Cannot Lose':'pink', 'At Risk': 'lightskyblue', 'Need Attention': 'darkturquoise', 'Champions':'gold' },
                                                title='',
                                                width=1000, height=500,
                                                )
        st.plotly_chart(fig_treemap_segment_customers)

        st.subheader('Comparison of Customer Segments')

        # Bar graph comparing the number of customers in each segment
        import plotly.colors

        pastel_colors = plotly.colors.qualitative.Pastel

        segment_customer_counts_bar = st.session_state['df_cluster_scaled_segment']['Customer_Segments'].value_counts()

        # Create a bar chart to compare segment counts
        fig_segment_customer_counts_bar = go.Figure(data=[go.Bar(x=segment_customer_counts_bar.index, y=segment_customer_counts_bar.values,
                                    marker=dict(color=pastel_colors))])

        # Set the color of the Champions segment as a different color
        champions_color = 'rgb(158, 202, 225)'
        fig_segment_customer_counts_bar.update_traces(marker_color=[champions_color if segment == 'Champions' else pastel_colors[i]
                                        for i, segment in enumerate(segment_customer_counts_bar.index)],
                        marker_line_color='rgb(8, 48, 107)',
                        marker_line_width=1, opacity=0.6)

        # Update the layout
        fig_segment_customer_counts_bar.update_layout(title='',
                        xaxis_title='RFM Segments',
                        yaxis_title='Number of Customers',
                        showlegend=False, width=1000, height=1000)

        st.plotly_chart(fig_segment_customer_counts_bar)

elif choice == 'Performance Evaluation': 
    
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from tabulate import tabulate
    
    if st.checkbox('Performance Evaluation of Method 1 - Segmentation based on combined RFM score.'):
    # Compute number of customers
        num_observations = len(st.session_state['df_rfm_score'])

        # Separate the features and the cluster labels
        rfm_X = st.session_state['df_rfm_score'].drop(['Value_Segment','RFM_Customer_Segments'], axis=1)
        rfm_segments = st.session_state['df_rfm_score']['Value_Segment']

        # Compute the metrics
        rfm_sil_score = silhouette_score(rfm_X, rfm_segments)
        rfm_calinski_score = calinski_harabasz_score(rfm_X, rfm_segments)
        rfm_davies_score = davies_bouldin_score(rfm_X, rfm_segments)

        # Create a table to display the metrics and the number of observations
        table_data = [
            ["Number of Observations", num_observations],
            ["Silhouette Score", rfm_sil_score],
            ["Calinski Harabasz Score", rfm_calinski_score],
            ["Davies Bouldin Score", rfm_davies_score]
        ]
        st.write("Number of Observations : ", num_observations)
        st.write("Silhouette Score : ", rfm_sil_score)
        st.write("Calinski Harabasz Score : ", rfm_calinski_score)
        st.write("Davies Bouldin Score : ", rfm_davies_score)
         
    if st.checkbox('Performance Evaluation of Method 2 - Segmentation individual Recency, Frequency and Monetary features'):

        # Compute number of customers
        num_observations = len(st.session_state['df_cluster_scaled_segment'])

        # Separate the features and the cluster labels
        X = st.session_state['df_cluster_scaled_segment'].drop(['Cluster', 'Customer_Segments', 'R_Score', 'F_Score', 'M_Score'],axis=1)
        clusters = st.session_state['df_cluster_scaled_segment']['Cluster']

        # Compute the metrics
        sil_score = silhouette_score(X, clusters)
        calinski_score = calinski_harabasz_score(X, clusters)
        davies_score = davies_bouldin_score(X, clusters)

        # Create a table to display the metrics and the number of observations
        table_data = [
            ["Number of Observations", num_observations],
            ["Silhouette Score", sil_score],
            ["Calinski Harabasz Score", calinski_score],
            ["Davies Bouldin Score", davies_score]
        ]
        st.write("Number of Observations : ", num_observations)
        st.write("Silhouette Score : ", sil_score)
        st.write("Calinski Harabasz Score : ", calinski_score)
        st.write("Davies Bouldin Score : ", davies_score) 

elif choice == 'Conclusion and Limitation': 
    st.write("""
    - Method 2 performs better and gives more accurate insight on customer segments.

    - K-means is sensitive to the initial placement of centroids. Different initializations may result in different final cluster assignments.

    - The choice of initial centroids can impact the convergence and final segmentation.

    - K-means assumes that clusters are spherical and equally sized. In reality, clusters may have different shapes, densities, or sizes. It may struggle with clusters that have complex geometries.
    
    - Although the accuracy of method 2 is more, since the data is unlabeled, the real-life scenario ma vary from what the model predicts.

    - Cannot cluster and predict the value and customer segment for one specific customer data, as the data points must be equal or more than the number of optimal cluster.

    """)    




def main():
    
    html_temp = """
    
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    # variance = st.text_input("Variance","Type Here")
    # skewness = st.text_input("skewness","Type Here")
    # curtosis = st.text_input("curtosis","Type Here")
    # entropy = st.text_input("entropy","Type Here")
    # result=""
    # if st.button("Predict"):
    #     result=predict_note_authentication(variance,skewness,curtosis,entropy)
    # st.success('The output is {}'.format(result))
    # if st.button("About"):
    #     st.text("Lets LEarn")
    #     st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
