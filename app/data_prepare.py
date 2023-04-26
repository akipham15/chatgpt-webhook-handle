import csv


with open('../data/train/fptqa.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        print(row[0])
        print(row[1])
        print('\n')
