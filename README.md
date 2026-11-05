# Bike Ride-Share User Analysis
**Goal:**

Increase revenue by converting more casual riders to members using a limited advertising budget by analyzing where and when to place targeted ads. 

**Audience:**

•	Manager responsible for the development of campaigns and initiatives to promote the bike-share program.
•	Analytics team (Me)
•	Executive Team – Detail-oriented executive who will decide whether to approve the recommended marketing program.


**Current State:**

Divvy currently relies on building general awareness and appealing to broad consumer segments. 
Has three tiers of passes: Single Ride, Full Day, and Annual Ride. 

Casual Riders: Anyone who purchases a single-ride pass or a full-day pass.

Annual Members: Purchase yearly memberships.

Finance Analysts say Annual Members are the most profitable and key to future growth. 

The manager notes that casual riders already know the program and have chosen Divvy for their mobility needs.

The manager's goal is to design a marketing strategy that converts casual riders into annual members.  Identify which stations casual riders use most and when they are most active to target advertising.  

**Ask:**

How do annual members use bikes differently from casual riders?

Where and when should Divvy place targeted ads to reach the most casual riders?

How can a machine learning model help the advertising department? 


What is the problem to solve?  
Identify which stations and when to place targeted ads to convert more casual riders into members. 

Business Task: Show how annual members use the bikes differently than casual riders and where and when to position targeted ads to convert casuals to members.  
 
Consider the Key Stakeholders: 
Marketing Manager. 
Detail-oriented Executives.


**Prepare:**

Guiding Questions: 

Where is the data located?
**ORIGINAL DATASET:** https://divvy-tripdata.s3.amazonaws.com/index.html
(Quarterly report)

Does the data ROCCC?

Reliability- 
Yes, it is a company result, but it would have to be inspected, cleaned, and organized for the purposes of this analysis

Original-
This is company data—first-party data from Divvy's archives. 

Comprehensive- 
It is not comprehensive because of privacy restrictions on customer data and the lack of customer IDs to identify individuals. 

Current – 
Data is not real-time. For planning next seasons, the data is recent enough

Cited –

Publicly available on Divvy's trip-data site and released under their data license for use
https://divvy-tripdata.s3.amazonaws.com/index.html 
https://divvybikes.com/data-license-agreement

Key Tasks: 
- Download the 2024-2026 monthly files
- Store raw files untouched; work only on copies
- Check the structure: row count, column names, column types
- Confirm the target column exists
- Look at quality: missing values and duplicates
- Set rules for outliers
- Identify errors and remove the rows
  
**Limitations of Data:**

What are the limits of the analysis? 

The data does not include individual riders' usage.  Therefore, the analysis will focus on aggregate groups rather than individual patterns. 

**Process**

Tools: Use Python to clean, organize, and visualize the data. 

Ensure Data Integrity: Keep a backup of the original raw data;
Work only on copies of the data.

**Steps to clean the data:**

-	Remove null spaces

-	Check for spelling errors.

-	Trim extra spaces

-	Format the time to be in the HH:MM: SS format

-	Create a new column called ride_length to track the average ride_length of each group by subtracting the started_at from the ended_at.

-	Ensure times are in minutes across the board

-	Filter columns with the same start and end station as round_trips and store them in a new column

-	Create a column called day_of_week and calculate the day of the week that each ride starts, 1= Sunday and 7= Saturday

-	Create a column for month/year to fit the wave model.

-	Convert months into sine and cosine values so the model treats the calendar as a circle after one-hot encoding the months 

-	Organize data into temporary tables

-	Create outlier_table for rides < 1 minute or > 120 minutes; the user probably forgot to return the bike. Compare which group falls mostly into these tables and hypothesize why.

**Verify Data is Clean and Ready to Analyze:**

-	Check for duplicate rows.

-	Check for null values

-	Check for the  appropriate length of values in each column

-	Check if start times and end times make sense (end times cannot be less than start times. 

-	Check for outliers like < 1 minute or > 120 minutes

-	Flag outliers for analysis, for instance, to determine who cancels more, who keeps the bike the longest, and add them to a new table.

- Run summary statistics like min, max, and  mean values

- Check for ride times that are impossibly short or long


**Document Cleaning Process:**

Use a Python Notebook to document the whole process and export it in a .py file

**Key Task:**
Check the data for errors

**Analyze**

What will I calculate: 

Duration and Frequency:
-	How long does each group ride?

-	How many trips do each group ride?

-	Which group rides for the longest time?

-	When does each group ride the most, what time of day, and what day of the week?


Location Patterns:
-	Which station is most popular for each group?

-	Which group ends at the same station the most (round_trips)

-	What percentage of rides are round-trip?

-	What percentage of rides are one-way? 

-	For one-way rides, average ride length per group. 

Outliers:
-	Which group falls the most into the outlier table?

Is the Data properly formatted?	
Suprises	
Trends	
Insights	

Key Task:
	Aggregate data so it is valuable and accessible

-	Organize and format your data

-	Perform calculations

-	Identify trends and relationships

-	 Create a machine learning algorithm to know when casual riders ride the most
  -	 Which stations casual riders start from the most
  -	 Run ads at those stations during peak casual rider season
  -	 Figure out when all riders ride the least to set up repair time
-  Test the model again with the 2026 data to estimate RMSE

Deliverable:
A description of all data sources.

Data is from Divvy quarterly reports. 

**Instructions to find data**

If using colab:

     from google.colab import drive
     drive.mount("/content/drive")

     import pandas as pd
     df = pd.read_parquet("/content/drive/MyDrive/Data Programmming Project Folder/Divvy_clean/divvy_clean_2024_2026.parquet")

if using vs code

     df = pd.read_parquet("data/divvy_clean_2024_2026.parquet")
	 
