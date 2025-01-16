import json



def load_json(file,base_path):
    """
    加载 JSON 文件并返回数据。

    参数:
        file: JSON 文件路径

    返回:
        JSON 文件数据
    """
    color_labels = {
        "red": 0,
        "green": 1,
        "blue": 2,
        "yellow": 3,
        "orange": 4,
        "purple": 5,
        "pink": 6,
        "black": 7,
        "white": 8,
        "gray": 9,
        "box_car": 10,
        "pickup_car": 11,
        "bus":12,
        "truck":13,
        "work_car_head":14,
        "work_car":15,
        "car":16,
        "suv":17,
        "fire_truck":18
    }
    image_paths=[]
    labels=[]
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for key, value in data.items():
        key = base_path+key
        image_paths.append(key)
        label = [0]*19
        for v in value:
            label[color_labels[v]]=1
        labels.append(label)

    return image_paths,labels

# if __name__ == "__main__":
#     base_path = r"data\jpg_data\\"
#     file = r"E:\multi_g\simplt_1\simplt\data\json_data\final.json"
#     image_paths,labels = load_json(file,base_path)
#     print(image_paths,labels)