# %% [markdown]
# # Divvy Data Cleaning
# - Loaded and combined the files 32files and 16,304,359 rows
# 
# - Removed duplicates 35 rows
# 
# - Removed rides with no end location 16,951
# 
# - Removed outliers and saved them in a new dataframe
# 
# - Remaining rows 15,851,351 (97.2% of original dataset)
# 
# **Reasoning** 
# 
# - Duplicates corrupt the data
# 
# - Kept rides with missing station names because they still count for the ride
# 
# - Removed rides with no end location because those are rides where the user probably forgot to return the bike to the station or end their rides properly.
# 
# - Rides under 1 minute are dropped because can't have a negative ride
# 
# - Rides over 180 minutes are dropped, casual rides rises with ride length over 180 are mostly docking error rather than real trips
# 
# **Column and Meaning**
# 
# ride_id - unique id of each ride
# 
# rideable_type - classic bike or electric
# 
# started_at/ended_at - start and end date and time
# 
# start_station_name and end_station_name - sometimes null may be blank for e-bikes
# 
# start_lat and end_lat - cannot be null, all rides most start and end loaction
# 
# member_casual - member type, member or casual
# 
# ride_length - added column to have the ride duration in minute
# 
# **Variables and what they hold**
# 
# files - the list of the 32 csv file paths
# 
# df - the dataframe
# 
# is_outlier - True/False with rides under a minute and over 180 minutes
# 
# - outliers - the removed outlier rides, kept for reference
# 
# 

# %%
import pandas as pd 
import numpy as np 
import glob

files = glob.glob("data/*.csv") # merge all files into a single folder
print(len(files))

# %%
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) # concat all 32 files into one
print(len(df))

# %%
df.head() # check the earliest data loaded

# %%
df.tail() # check that the most recent data loaded

# %%
first_cols = list(pd.read_csv(files[0], nrows=0).columns)

for f in files:
    cols = list(pd.read_csv(f, nrows=0).columns)
    if cols != first_cols:
        print("DIFFERENT:", f)
        print(cols)

print("Check finished")  # ensuring all column names are the same through the years

# %%
is_dupe = df.duplicated(keep=False) # find duplicated row, marking every copy so we can see the original and twin side by side
dupes = df[is_dupe] # keep the only rows that have duplicates
dupes.sort_values("ride_id").head(40) # align the matching copies with the next so easy to compare

# %%
rows_before = len(df) # number of row before dropping duplicate
df = df.drop_duplicates() # number of rows after dropping duplicate
print("Duplicate removed", rows_before - len(df)) # print how many rows were dropped

# %%
print(len(df))
df.isna().sum() # see the total number of missssig cells per column

# %%
#drop rows with no start or end lat because every bike not be picked up or returned to a station. 

row_before_drop = len(df)
df = df.dropna(subset=['end_lat', 'end_lng'])
print('rows removed', row_before_drop - len(df))

# %%
print(len(df))

# %%
#craete  a new column called ride_length and see the mean and check for rows that greatly exceed the ride length

df['started_at'] = pd.to_datetime(df["started_at"], format='mixed') # change to datetime from string for calculations
df['ended_at'] = pd.to_datetime(df['ended_at'], format='mixed')

df['ride_length'] = (df['ended_at']- df['started_at']).dt.total_seconds() / 60 # time in minutes

# %%
# see the mean ride time
pd.set_option("display.float_format", "{:,.2f}".format) # so its set to regular notation
df['ride_length'].describe().apply(lambda x: f"{x :,.2f}") #show decimals instead of scientific notation and add commas to make it more readable


# %%
print("Negative or zero:", (df["ride_length"] <= 0).sum())
print("Under 1 min:", (df["ride_length"] < 1).sum())
print("Over 60 min:", (df["ride_length"] > 60).sum())
print("Over 120 min:", (df["ride_length"] > 120).sum())

# %%
df["member_casual"].value_counts(normalize=True)

# %%
# compare lenght of rides between casual and member to know when to cut of outliers
valid_long = df[df["ride_length"] > 1]
bins = [0, 15, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 1600]
long_bins = pd.cut(valid_long["ride_length"], bins=bins)

pd.crosstab(long_bins, valid_long["member_casual"])

# %%
before_outliers = len(df)

is_outlier = (df["ride_length"] < 1) | (df['ride_length'] > 180) # outliers
outliers = df[is_outlier] # save the bad rides
df = df[~is_outlier] #keep everything else

print(f"Before: {before_outliers:,}")
print(f"Outliers:  {len(outliers):,}")
print(f"Clean: {len(df):,}")

if len(outliers) + len(df) == before_outliers:
    print("All rows accounted for")
    
else:
    print("Error")


