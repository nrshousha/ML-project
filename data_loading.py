# data loading
def read_csv(path):
    data = []
    with open("/Users/Noura/Desktop/work/college/level 002/second sem/ML/Titanic project/data set/train.csv", 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        for line in f:
            row = []
            value = ''
            inside_quotes = False
            for c in line:
                if c == '"':
                    inside_quotes = not inside_quotes
                elif c == ',' and not inside_quotes:
                    row.append(value)
                    value = ''
                else:
                    value += c
            row.append(value.strip())
            data.append(row)
    return header, data