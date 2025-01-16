import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import json

# 定义标签****************************************************************************************************************
LABELS = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "black", "white", "gray"]

# 初始化标签字典
image_labels = {}
json_path = ""  # 全局变量，用于存储当前打开的 JSON 文件路径

# 创建主窗口
root = tk.Tk()
root.title("Image Classification Label Tool")
root.geometry("800x600")

# 显示图像的标签
image_label = tk.Label(root)
image_label.pack(pady=20)

# 添加文件名显示标签
filename_label = tk.Label(root, text="", font=("Arial", 10))
filename_label.pack(pady=5)

# 标签显示区域
label_display = tk.Label(root, text="Labels: ", font=("Arial", 12))
label_display.pack(pady=10)

# 当前图片索引
current_image_index = 0
image_paths = []


# 加载图片
def load_image(image_path):
    global current_image_index
    image = Image.open(image_path)
    image = image.resize((400, 400))  # 缩放图像以适应界面
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

    # 显示当前图片文件名
    current_image = os.path.basename(image_path)
    filename_label.config(text=f"Current image: {current_image}")

    # 清除复选框的选中状态
    for var in label_vars.values():
        var.set(False)

    # 获取当前图像的文件名
    current_image = os.path.basename(image_paths[current_image_index])

    # 加载当前图像的标签（如果有的话）
    if current_image in image_labels:  # 直接使用文件名作为键
        found_labels = image_labels[current_image]
        for label in found_labels:
            if label in label_vars:
                label_vars[label].set(True)
        label_display.config(text="Labels: " + ", ".join(found_labels))
    else:
        label_display.config(text="Labels: None")


# 浏览文件夹并加载图像
def browse_folder():
    global image_paths
    folder_path = filedialog.askdirectory()  # 选择文件夹
    if folder_path:
        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                       f.endswith(('png', 'jpg', 'jpeg', 'bmp'))]
        if image_paths:
            load_image(image_paths[current_image_index])
            next_button.config(state=tk.NORMAL)
            prev_button.config(state=tk.NORMAL)
        else:
            messagebox.showwarning("No Images", "No images found in the selected folder.")


# 下一张图像
def next_image():
    global current_image_index
    if current_image_index < len(image_paths) - 1:
        current_image_index += 1
        load_image(image_paths[current_image_index])
    else:
        messagebox.showinfo("End", "You have reached the last image.")


# 上一张图像
def prev_image():
    global current_image_index
    if current_image_index > 0:
        current_image_index -= 1
        load_image(image_paths[current_image_index])
    else:
        messagebox.showinfo("Start", "You are at the first image.")


# 更新图像的标签
def update_labels():
    # 获取当前选中的预定义标签
    selected_predefined_labels = [label for label, var in label_vars.items() if var.get()]

    # 只获取文件名，不要路径
    current_image_filename = os.path.basename(image_paths[current_image_index])

    # 找到与当前图像文件名匹配的 JSON 键
    matched_key = None
    for key in image_labels:
        if key == current_image_filename:  # 直接比较文件名
            matched_key = key
            break

    # 如果没有找到匹配的键，使用当前图像文件名作为新键
    if not matched_key:
        matched_key = current_image_filename
        image_labels[matched_key] = []

    # 获取现有的标签
    existing_labels = image_labels[matched_key]
    
    # 保留不在预定义标签列表中的标签
    other_labels = [label for label in existing_labels if label not in LABELS]
    
    # 合并其他标签和当前选中的预定义标签
    updated_labels = other_labels + selected_predefined_labels
    
    # 更新标签
    image_labels[matched_key] = updated_labels

    # 更新显示
    label_display.config(text="Labels: " + ", ".join(updated_labels))

    # 保存更新后的标签到文件
    if json_path:  # 确保有JSON文件路径
        try:
            with open(json_path, "w") as f:
                json.dump(image_labels, f, indent=4)
            messagebox.showinfo("Saved", f"Labels for image {current_image_filename} updated successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while saving the labels: {e}")
    else:
        messagebox.showwarning("No Save File", "Please use 'Save All Labels to File' first to create a JSON file.")


# 保存标签
def save_labels():
    save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
    if save_path:
        global json_path, image_labels
        json_path = save_path  # 保存文件路径
        
        # 创建新的空字典，不使用之前可能加载的标签
        image_labels = {}
        
        try:
            # 创建新的JSON文件
            with open(json_path, "w") as f:
                json.dump(image_labels, f, indent=4)
            messagebox.showinfo("Created", f"New label file created at {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while creating the new label file: {e}")


# 从JSON文件加载标签
def load_labels_from_json():
    global image_labels, json_path
    json_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if json_path:
        try:
            with open(json_path, "r") as f:
                image_labels = json.load(f)

            # 加载首个图像
            load_image(image_paths[current_image_index])
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while loading the labels: {e}")


# 为每个标签创建复选框
label_vars = {}
label_frame = tk.Frame(root)
label_frame.pack(pady=10)

for label in LABELS:
    var = tk.BooleanVar()
    label_vars[label] = var
    checkbox = tk.Checkbutton(label_frame, text=label, variable=var)
    checkbox.pack(side=tk.LEFT)

# 创建按钮
browse_button = tk.Button(root, text="Browse Folder", command=browse_folder)
browse_button.pack(pady=10)

prev_button = tk.Button(root, text="Previous Image", command=prev_image, state=tk.DISABLED)
prev_button.pack(side=tk.LEFT, padx=20)

next_button = tk.Button(root, text="Next Image", command=next_image, state=tk.DISABLED)
next_button.pack(side=tk.LEFT, padx=20)

save_button = tk.Button(root, text="Save Labels", command=update_labels)
save_button.pack(pady=10)

save_to_file_button = tk.Button(root, text="Save All Labels to File", command=save_labels)
save_to_file_button.pack(pady=10)

load_json_button = tk.Button(root, text="Load Labels from JSON", command=load_labels_from_json)
load_json_button.pack(pady=10)

# 启动主循环
root.mainloop()
