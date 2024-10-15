from os import listdir
from os.path import join, isfile
import csv

files = [f for f in listdir('results_shortest_europe/loss') if isfile(join('results_shortest_europe/loss',f))]

f_save = open("results_shortest_europe/loss_agragated.csv", "w")
for f in files:
    data = []
    with open(join('results_shortest_europe/loss', f), 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';')
        for row in csv_reader:
            line = []   
            for word in row:
                if word.replace('.','',1).isdigit():
                    line.append(float(word))
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


files = [f for f in listdir('results_shortest_europe/results') if isfile(join('results_shortest_europe/results',f))]

f_save = open("results_shortest_europe/results_agragated.csv", "w")
f_save.write('Activation')
f_save.write(';')
f_save.write('Loss')
f_save.write(';')
f_save.write('Ratio')
f_save.write(';')
f_save.write('Detour')
f_save.write(';')
f_save.write('Detour_Ratio')
f_save.write(';')
f_save.write('d_time_mean')
f_save.write(';')
f_save.write('a_time_mean')
f_save.write(';')
f_save.write('o_time_mean')
f_save.write(';')
f_save.write('\n')

for f in files:
    with open(join('results_shortest_europe/results', f), 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';')
        N_path_found = 0
        N_path_n = 0
        detour = 0
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
                    detour += float(row[3]) - float(row[1])
                    detour_ratio = detour/float(row[1])
                    d_time_sum += float(row[4])
                    a_time_sum += float(row[5])
                    o_time_sum += float(row[6])

            i += 1
        path_found_ratio = N_path_found/N_path_n*100
        detour_mean = detour/N_path_found
        detour_ratio_mean = detour_ratio/N_path_found
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
        f_save.write(str(detour_mean))
        f_save.write(';')
        f_save.write(str(detour_ratio_mean))
        f_save.write(';')
        f_save.write(str(d_time_mean))
        f_save.write(';')
        f_save.write(str(a_time_mean))
        f_save.write(';')
        f_save.write(str(o_time_mean))
        f_save.write(';')

        f_save.write('\n')

f_save.close()