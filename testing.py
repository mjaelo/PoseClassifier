# program w pythonie
def testing():
    liczba = int(input("podaj liczbe"))
    level = 0
    for i in range(liczba + 1):
        padding_middle = ''
        padding_start = ''
        padding_start_nr = liczba - i
        for j in range(level):
            padding_middle += '__'
        for k in range(padding_start_nr):
            padding_start += ' '
        print(padding_start + '/' + padding_middle + '\\')
        level += 1


# ---/\
# --/--\
# -/----\

def findMissing(tablica):
    for i in range(len(tablica)):
        found = 0
        for nr in tablica:
            if nr == i:
                found = 1
        if found == 0:
            return i


if __name__ == '__main__':
    # testing()
    tablica = [6, 0, 5, 2, 1, 4, 3]
    i = findMissing(tablica)
    print(i)
