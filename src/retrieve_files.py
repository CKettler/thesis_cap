import os

class retrieve_files:
    def __init__(self, directory_path, variables=None):
        self.directory_path = directory_path

    def get_xls_files(self):
        dict_of_files_GPS = {}
        dict_of_files_results = {}
        for (dirpath, dirnames, filenames) in os.walk(self.directory_path):
            for filename in filenames:
                if filename.endswith('.xls'):
                    if 'gps' in filename.lower():
                        dict_of_files_GPS[filename] = os.sep.join([dirpath, filename])
                    elif 'results' in filename.lower():
                        dict_of_files_results[filename] = os.sep.join([dirpath, filename])
                    else:
                        continue
                elif filename.endswith('.xlsx'):
                    if 'gps' in filename.lower():
                        dict_of_files_GPS[filename] = os.sep.join([dirpath, filename])
                    elif 'results' in filename.lower():
                        dict_of_files_results[filename] = os.sep.join([dirpath, filename])
                    else:
                        continue
        return dict_of_files_GPS, dict_of_files_results

    def get_csv_files(self):
        dict_of_files_GPS = {}
        for (dirpath, dirnames, filenames) in os.walk(self.directory_path):
            for filename in filenames:
                if filename.endswith('.csv'):
                    dict_of_files_GPS[filename] = os.sep.join([dirpath, filename])
        return dict_of_files_GPS
