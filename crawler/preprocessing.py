import time

PATH_FILES = "../data/raw-vnexpress-contents-update.txt"

final_output = []
with open(PATH_FILES, 'r',  encoding='utf-8') as f:
    for raw_line in f:
        raw_line = raw_line.lower().strip()
        line_split = raw_line.split('.')
        for line in line_split:
            final_output.append(line.strip())

with open('../data/preprocessing-vnexpress-contents-update.txt', 'w') as f:
    for line in final_output:
        f.writelines(line + '\n')