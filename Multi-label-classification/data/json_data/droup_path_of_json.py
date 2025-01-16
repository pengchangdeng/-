def format_json_with_style(input_json):
    formatted_lines = ['{']
    for i, (key, value) in enumerate(input_json.items()):
        # 获取文件名
        filename = key.split('\\')[-1]
        
        # 构建每个条目的格式
        formatted_lines.append(f'    "{filename}": [')
        
        # 保持原有的标签
        for j, label in enumerate(value):
            formatted_lines.append(f'        "{label}"' + (',' if j < len(value) - 1 else ''))
            
        formatted_lines.append('    ]' + (',' if i < len(input_json) - 1 else ''))
        
    formatted_lines.append('}')
    return '\n'.join(formatted_lines)

# 使用示例
import json

# 读取输入文件
with open(r'E:\multi_g\simplt_1\simplt\data\json_data\final.json', 'r') as f:
    data = json.load(f)

# 格式化JSON
formatted_output = format_json_with_style(data)

# 保存到输出文件
with open(r'E:\multi_g\simplt_1\simplt\data\json_data\final_output.json', 'w') as f:
    f.write(formatted_output)

print("Processing completed!")