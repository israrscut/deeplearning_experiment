# first line: 5
@mem.cache
def get_data():
    data = load_svmlight_file("housing_scale.txt")
    return data[0], data[1]
