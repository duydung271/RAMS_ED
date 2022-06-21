a = [[1,2,3], [4,5,6,7], [8.9]]

def flatten(items):
    all_items =  [y for x in items for y in x]
    return all_items

if __name__ == '__main__':

    print(flatten(a))