import os

numFiles = 154
file_names = os.listdir('Data/testVisualization')#List of file names in the directory
file_names = sorted(file_names, key=lambda item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item))
file_names = file_names[0:numFiles]
print(file_names)