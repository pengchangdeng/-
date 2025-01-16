import json
import os
from collections import defaultdict
import traceback

def merge_json_files(json_folder_path, output_path):
    try:
        # 检查文件夹是否存在
        if not os.path.exists(json_folder_path):
            print(f"Error: Folder not found: {json_folder_path}")
            return
            
        print(f"Scanning folder: {json_folder_path}")
        
        # 使用defaultdict来存储合并后的标签
        merged_labels = defaultdict(set)
        
        # 遍历文件夹中的所有json文件
        json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]
        print(f"Found {len(json_files)} JSON files")
        
        for filename in json_files:
            print(f"Processing file: {filename}")
            json_path = os.path.join(json_folder_path, filename)
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 遍历JSON文件中的每个条目
                for key, labels in data.items():
                    # 获取最后一个反斜杠后的文件名
                    image_filename = key.split('\\')[-1]
                    # 将标签添加到对应文件名的集合中，保持原有标签不变
                    merged_labels[image_filename].update(labels)
            
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                print("Detailed error:")
                print(traceback.format_exc())
        
        # 将集合转换为列表，准备保存
        final_dict = {k: list(v) for k, v in merged_labels.items()}
        
        print(f"Total unique images processed: {len(final_dict)}")
        
        # 保存合并后的结果
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_dict, f, indent=4)
            print(f"Successfully merged labels to {output_path}")
        except Exception as e:
            print(f"Error saving merged file: {str(e)}")
            print("Detailed error:")
            print(traceback.format_exc())

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        print("Detailed error:")
        print(traceback.format_exc())

if __name__ == "__main__":
    try:
        # 使用正确的路径
        json_folder = r"E:\multi_g\simplt_1\simplt\data\json_data"  # JSON文件所在的文件夹
        output_file = r"E:\multi_g\simplt_1\simplt\data\json_data\merged_labels.json"  # 合并后的输出文件
        
        print("Starting merge process...")
        merge_json_files(json_folder, output_file)
        print("Process finished")
    except Exception as e:
        print(f"Main process error: {str(e)}")
        print("Detailed error:")
        print(traceback.format_exc())

    # 等待用户输入，防止窗口立即关闭
    input("Press Enter to exit...")
