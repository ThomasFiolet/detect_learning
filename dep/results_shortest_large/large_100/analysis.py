from os import listdir
from os.path import join, isfile
import csv

files = [f for f in listdir('results_shortest_large/large_100/loss') if isfile(join('results_shortest_large/large_100/loss',f))]

f_save = open("results_shortest_large/large_100/loss_agragated.csv", "w")
for f in files:
    data = []
    with open(join('results_shortest_large/large_100/loss', f), 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';')
        
        for row in csv_reader:
            line = []
            i = 0
            for word in row:
                if i != 0:
                    if word.replace('.','',1).isdigit():
                        line.append(float(word))
                i += 1
            data.append(line)

        agg = [sum(x) for x in zip(*data)]
        splt = f.split('.', 1)[0]
        splt = splt.split('_')
        f_save.write(splt[1])
        f_save.write(';')
        f_save.write(splt[2])
        f_save.write(';')
        for s in agg:
            f_save.write(str(s))
            f_save.write(';')
        f_save.write('\n')

f_save.close()


files = [f for f in listdir('results_shortest_large/large_100/results') if isfile(join('results_shortest_large/large_100/results',f))]

f_save = open("results_shortest_large/large_100/results_agragated.csv", "w")
f_save.write('Activation')
f_save.write(';')
f_save.write('Loss')
f_save.write(';')
f_save.write('Ratio')
f_save.write(';')
f_save.write('MSE')
f_save.write(';')
f_save.write('d_time_mean')
f_save.write(';')
f_save.write('a_time_mean')
f_save.write(';')
f_save.write('o_time_mean')
f_save.write(';')
f_save.write('\n')

for f in files:
    with open(join('results_shortest_large/large_100/results', f), 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';')
        N_path_found = 0
        N_path_n = 0
        sum_square_of_diff = 0
        path_found_ratio = 0
        d_time_sum = 0
        a_time_sum = 0
        o_time_sum = 0
        i = 0
        for row in csv_reader:
            if i != 0:
                N_path_n += 1
                if float(row[3]) > 0: 
                    N_path_found += 1
                    sum_square_of_diff += float(row[3]) - float(row[1])
                    d_time_sum += float(row[4])
                    a_time_sum += float(row[5])
                    o_time_sum += float(row[6])
            i += 1
        path_found_ratio = N_path_found/N_path_n*100
        mean_of_diff = sum_square_of_diff/N_path_found
        d_time_mean = d_time_sum/N_path_found
        a_time_mean = a_time_sum/N_path_found
        o_time_mean = o_time_sum/N_path_found

#Path name
#Dijkstra distance
#A Star distance
#Ours distance
#Dijkstra time
#A Star time
#Ours time

        splt = f.split('.', 1)[0]
        splt = splt.split('_')
        f_save.write(splt[1])
        f_save.write(';')
        f_save.write(splt[2])
        f_save.write(';')

        f_save.write(str(path_found_ratio))
        f_save.write(';')
        f_save.write(str(mean_of_diff))
        f_save.write(';')
        f_save.write(str(d_time_mean))
        f_save.write(';')
        f_save.write(str(a_time_mean))
        f_save.write(';')
        f_save.write(str(o_time_mean))
        f_save.write(';')

        f_save.write('\n')

f_save.close()