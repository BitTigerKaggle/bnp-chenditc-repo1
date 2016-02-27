import csv

# Get sample format
sample_reader = csv.reader(open('sample_submission.csv'), delimiter=',')

id_array = []
score = []
for row in sample_reader:
    id_array.append(row[0])
    score.append(row[1])

# Get prediction from csv
aws_result_reader = csv.reader(open('aws_result_1.csv'), delimiter=',')
aws_score = []
for row in aws_result_reader:
    aws_score.append(row[1])
aws_score[0] = 'PredictedProb'

# Merge results and generate new submitssion file
for i in range(len(id_array)):
    if i != 0:
        aws_score[i] = float(aws_score[i])
    print id_array[i] + "," + str(aws_score[i])

