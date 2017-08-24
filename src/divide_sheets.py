import pandas as pd
from xlrd import open_workbook
import re
import numpy as np

dir_path = "../data/Excel_from_PDF/"
dir_path_to = "../data/Racedata/"
# years = ["2015"]
years = ["2013","2014","2016"]
comps = ["ECH", "WC1", "WC2", "WC3", "WCH"]
extensions = ["Results.xlsx","GPS.xlsx"]
for extension in extensions:
    for year in years:
        for comp in comps:
            filename = year + " " + comp + " " + extension
            print("[Processing %s]" % filename)
            try:
                wb = open_workbook(dir_path+filename)
            except:
                print("[%s is no valid filename]" % filename)
                continue
            no_sheets = len(wb.sheets())
            for sheet_no in range(77, no_sheets):
                print("[Sheet %s out of %s sheets]" % (sheet_no, no_sheets))
                df = pd.read_excel(dir_path+filename, sheetname=sheet_no, header=None)
                if "World Best Time:" == df.iloc[2,0]:
                    new_line_count = 0
                    nr_columns = df.shape[1]
                    new_df = pd.DataFrame(index=range(6), columns=df.columns)
                    new_df = new_df.fillna(np.nan)
                    for col_nr in range(nr_columns):
                        # value1 = df.iloc[0,col_nr]
                        # value2 = df.iloc[1, col_nr]
                        if type(df.iloc[0,col_nr]) == unicode:
                            if 'Results' in str(df.iloc[0,col_nr]):
                                result = df.iloc[0,col_nr]
                                df.iloc[0,col_nr] = np.nan
                            elif len(list(df.iloc[0,col_nr].decode("utf-8"))) == 4:
                                boattype = df.iloc[0, col_nr]
                                df.iloc[0, col_nr] = np.nan
                        if type(df.iloc[1,col_nr]) == unicode:
                            info_list = df.iloc[1, col_nr].split("\n")
                            if len(info_list) == 2 and new_line_count == 0:
                                value = df.iloc[1, col_nr]
                                [boat_description, date] = value.splitlines()
                                df.iloc[1,col_nr] = np.nan
                                new_line_count += 1
                            elif len(info_list) == 2 and new_line_count == 1:
                                value = df.iloc[1, col_nr]
                                [raceround , race] = value.splitlines()
                                df.iloc[1,col_nr] = np.nan
                                new_line_count += 1
                    df = pd.concat([df[:2], new_df.rename(columns=dict(zip(new_df.columns, df.columns))), df[2:]])
                    df = df.reset_index(drop=True)
                    df.iloc[2,0] = result
                    df.iloc[3, 0] = boat_description
                    df.iloc[4, 0] = date
                    df.iloc[5, 0] = boattype
                    df.iloc[6, 0] = raceround
                    df.iloc[7, 0] = race
                else:
                    boattype = df.iloc[5, 0]
                    raceround = df.iloc[6, 0]
                # print('raceround: %s' % raceround)
                # print('boattype: %s' % boattype)
                if '/' in raceround:
                    raceround = re.sub('/', '',raceround)
                    if raceround == "SAB 1":
                        raceround = "S1"
                    elif raceround == "SAB 2":
                        raceround = "S2"
                    elif raceround == "SCD 1":
                        raceround = "S3"
                    elif raceround == "SCD 2":
                        raceround = "S4"
                    elif raceround == "SEF 1":
                        raceround = "S5"
                    elif raceround == "SEF 2":
                        raceround = "S6"
                    elif raceround == "QAD 1" or raceround == "QAD 2" or raceround == "QAD 3" or raceround == "QAD 4":
                        raceround = re.sub("AD ", "", raceround)
                year = filename[:4]
                comp = filename[5:9]
                if 'GPS' in filename:
                    new_filename = year + ' - ' + comp + ' - ' + boattype + ' - ' + raceround + ' - ' + 'GPS.xlsx'
                elif 'Results' in filename:
                    new_filename = year + ' - ' + comp + ' - ' + boattype + ' - ' + raceround + ' - ' + 'Results.xlsx'
                df.to_excel(dir_path_to + str(year) + "/" + new_filename, header=False, index=False)
