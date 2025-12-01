import cv2
import numpy as np
import matplotlib.pyplot as plt

def is_box_inside(inner, outer):
    """
    判断 inner 框是否完全位于 outer 框内部
    格式: (x, y, w, h)
    """
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    return (ix >= ox) and (iy >= oy) and (ix + iw <= ox + ow) and (iy + ih <= oy + oh)

def draw_precise_boxes(image_path):
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"未找到图片: {image_path}")
        return

    result_img = img.copy()
    
    # 图像预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 使用 OTSU 二值化，自动寻找最适合区分墨迹和纸张的阈值
    # 这一步比固定阈值更准，能更好得提取字迹
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 使用【闭运算】连接断开的笔画
    # 定义核心距离值（Kernel_size）：
    # (10, 10) 这是一个关键参数，该参数是防止汉字部首被识别成多个汉字的关键参数，避免汉字被过度分割
    # 算法实现：如果两个笔画距离在 10 像素以内，就认为它们属于同一个字，把它们连起来。
    # 注意：这个值如果太大，会把上下两个字连起来；如果太小，左右结构的字会分家，这里需要自行调整  
    kernel_size = (12, 12) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # MORPH_CLOSE = 先膨胀后腐蚀。能闭合内部小孔和近距离的断裂，但保持轮廓大小基本不变。
    closed_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 收集所有候选框
    boxes = []
    min_area = 100 # 过滤掉噪点（太小的点）
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

    # 除嵌套 (Nesting Removal) 如果一个框在另一个框里面，只保留大的，删掉小的
    
    # 先按面积从大到小排序，确保先处理大框
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    
    keep = [True] * len(boxes)
    for i in range(len(boxes)):
        if not keep[i]: continue
        for j in range(i + 1, len(boxes)):
            if keep[j] and is_box_inside(boxes[j], boxes[i]):
                keep[j] = False # 标记为删除（内部的小框）

    final_boxes = [boxes[i] for i in range(len(boxes)) if keep[i]]

    # 绘制与着色
    count_long = 0
    count_square = 0
    count_flat = 0

    for (x, y, w, h) in final_boxes:
        # 在原图上绘制，注意：这里我们用的是原始坐标
        # 因为 closed_img 只是用来找位置，并没有改变坐标系
        
        aspect_ratio = h / float(w)
        
        if aspect_ratio > 1.2:
            color = (0, 0, 255) # 红 (长)
            count_long += 1
            label = "L"
        elif aspect_ratio < 0.8:
            color = (255, 0, 0) # 蓝 (扁)
            count_flat += 1
            label = "F"
        else:
            color = (0, 255, 0) # 绿 (方)
            count_square += 1
            label = "S"

        # 绘制矩形
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)

    # 输出统计与保存
    print("--- 处理完成 ---")
    print(f"保留框总数: {len(final_boxes)}")
    print(f"🔴 长形字 (>1.2): {count_long}")
    print(f"🟢 方形字 (0.8-1.2): {count_square}")
    print(f"🔵 扁形字 (<0.8): {count_flat}")

    output_path = "precise_calligraphy_boxes.jpg"
    cv2.imwrite(output_path, result_img)
    
    # 显示结果
    plt.figure(figsize=(12, 18))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # 处理的图片文件
    draw_precise_boxes('./20251201203533_88_145.jpg')
