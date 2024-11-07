import os
import sys

def list_data_sources(directory, file_type):
    # Check the supported file type
    if file_type not in ['fit', 'fits.gz']:
        raise ValueError("Unsupported file type. Use 'fit' or 'fits.gz'.")

    # List all files in the specified directory
    all_files = os.listdir(directory)
    
    # Filter files based on the specified extension
    if file_type == 'fit':
        files = [f for f in all_files if f.endswith('.fit')]
    elif file_type == 'fits.gz':
        files = [f for f in all_files if f.endswith('.fits.gz')]
    
    # Remove the extensions from the found files
    files_no_ext = [os.path.splitext(f)[0] for f in files]
    
    # For .fits.gz files, also remove the .gz extension
    if file_type == 'fits.gz':
        files_no_ext = [f[:-4] for f in files_no_ext]
    
    return files_no_ext

if __name__ == '__main__':
    # get data_directory and file_type from command line
    data_directory = sys.argv[1]
    file_type = sys.argv[2]
    files_list = list_data_sources(data_directory, file_type)
    # Name a file with the directory name and the file type
    # Get the last part of the directory name
    dir_name = data_directory.split('/')[-1]
    file_name = dir_name + '_' + file_type + '.txt'
    print(dir_name)
    print(file_name)
    with open(file_name, 'w') as f:
        for item in files_list:
            f.write("%s\n" % item)