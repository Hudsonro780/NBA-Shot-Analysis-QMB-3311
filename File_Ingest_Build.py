# ---- Header -----

# Developer - Robert Hudson

# PURPOSE:

# to prepare datasets for class on 4/23/26

# dynamically load all csv files in location and create a master csv file for all NBA 
# shot data across seasons.

# -------------------------------------------------------------------------------- #

# ---- Packages ----
import os
import pandas as pd
import csv
import re

# ---- Cofig ----
#1990s files
folder_90s = r"C:\Users\Rober\Desktop\Coding Projects\temp_NBA\1990s"
#2000s files
folder_00s = r"C:\Users\Rober\Desktop\Coding Projects\temp_NBA\2000s"
# master output
output = r"C:\Users\Rober\Desktop\Coding Projects\temp_NBA"


Master_File = pd.DataFrame()

# ---- Ingest System ----

def ingest_system(folder_dir):

    files_loaded = []

    for filename in os.listdir(folder_dir):
        if filename.endswith(".csv"):
            year_match = re.search(r'(\d{2})_(\d{2})', filename)

            if year_match:
                y1 = int(year_match.group(1))
                y2 = int(year_match.group(2))

                y1_4digits = 1900 + y1 if y1 >= 90 else y1+2000
                y2_4digits = y1_4digits + 1

                file_season = str(y1_4digits) + "-"+str(y2_4digits)

                temp = pd.read_csv(os.path.join(folder_dir,filename))
                temp["Season"] = file_season

                print("File with season: ", file_season, "is loaded")

                files_loaded.append(temp)

                print("File with season: ", file_season," is added\n------------\n")

    return(files_loaded)

files = ingest_system(folder_90s) + ingest_system(folder_00s)

Master_File = pd.concat(files, ignore_index= True)

event_count = len(Master_File)

made_shots = len(Master_File[Master_File['EVENT_TYPE'] == "Made Shot"])
missed_shots = len(Master_File) - made_shots
most_attempts = Master_File.groupby('PLAYER_NAME').size().idxmax()

total_percent = round((made_shots/(event_count))*100,2)
print("Robert Hudson, The total field goal percentage = ", total_percent,"%.\n")

shot_percent = Master_File.groupby('PLAYER_NAME')['EVENT_TYPE'].apply(lambda x: (x=="Yes").mean()*100)

# Note total shooting percentage for the data is ~45.4%

# Not needed anymore
# Master_File.to_csv(os.path.join(output, "Master_NBA.csv"))

print("\n------------ File is created ---------------\n\nThere were ", event_count, " Shot attempts.")
print("\nThere were ", made_shots,"made.\nThere were ", missed_shots,"missed.\n\nThe trigger happy shooter was", most_attempts,".\n\n-----------------------")
