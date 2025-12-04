import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def is_box_inside(inner, outer):
    """
    判断 inner 框是否完全位于 outer 框内部
    格式: (x, y, w, h)
    """
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    return (ix >= ox) and (iy >= oy) and (ix + iw <= ox + ow) and (iy + ih <= oy + oh)

def get_adaptive_kernel_size(binary_img, multiplier=3.5):
    """
    核心算法：根据图像内容的笔画宽度，动态计算最佳闭运算核大小。
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stroke_widths = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area > 10 and perimeter > 10:
            width = 2 * area / perimeter
            stroke_widths.append(width)
    
    if not stroke_widths:
        return (10, 10)

    median_width = np.median(stroke_widths)
    k_size = int(median_width * multiplier)
    k_size = max(3, k_size) 
    
    print(f"📊 [自适应分析] 估算笔画宽度: {median_width:.2f} px")
    print(f"🔧 [自适应分析] 动态设定 Kernel Size: ({k_size}, {k_size})")
    
    return (k_size, k_size)

def process_calligraphy(image_path):
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 未找到图片: {image_path}")
        return

    print(f"📂 开始处理文件: {image_path}")
    
    # 准备两张画布：一张画框，一张画多边形
    analyzer_img = img.copy() # 用于 _analyzer.jpg
    shape_img = img.copy()    # 用于 _shape.jpg
    
    # 图像预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 适应获取 Kernel Size
    adaptive_kernel = get_adaptive_kernel_size(binary)
    
    # 形态学闭运算 (连接笔画)
    closed_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, 
                                  cv2.getStructuringElement(cv2.MORPH_RECT, adaptive_kernel))

    # 查找轮廓
    contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 收集候选数据
    # 这里同时保存 轮廓(contour) 和 边界框(box)
    candidates = []
    min_area = adaptive_kernel[0] * adaptive_kernel[1] * 2 
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            candidates.append({
                'cnt': cnt,      # 原始轮廓数据 (用于画多边形)
                'box': (x, y, w, h), # 矩形数据 (用于画框和去嵌套)
                'area': area     # 面积 (用于排序)
            })

    # 去除嵌套 (Nesting Removal)
    # 按面积从大到小排序
    candidates.sort(key=lambda c: c['area'], reverse=True)
    
    keep = [True] * len(candidates)
    for i in range(len(candidates)):
        if not keep[i]: continue
        for j in range(i + 1, len(candidates)):
            # 检查 j 是否在 i 内部
            if keep[j] and is_box_inside(candidates[j]['box'], candidates[i]['box']):
                keep[j] = False 

    # 过滤后的最终列表
    final_candidates = [candidates[i] for i in range(len(candidates)) if keep[i]]

    # 绘制逻辑
    count_long = 0
    count_square = 0
    count_flat = 0

    for item in final_candidates:
        # --- 绘制矩形框 (_analyzer) ---
        x, y, w, h = item['box']
        aspect_ratio = h / float(w)
        
        if aspect_ratio > 1.2:
            color = (0, 0, 255) # 红 (长)
            count_long += 1
        elif aspect_ratio < 0.8:
            color = (255, 0, 0) # 蓝 (扁)
            count_flat += 1
        else:
            color = (0, 255, 0) # 绿 (方)
            count_square += 1
            
        cv2.rectangle(analyzer_img, (x, y), (x + w, y + h), color, 2)

        # --- 绘制围合多边形 (_shape) ---
        # 使用 Convex Hull (凸包) 算法
        # hull 是一组点，代表了包围该轮廓的最小凸多边形
        cnt = item['cnt']
        hull = cv2.convexHull(cnt)
        
        # 参数说明：画板, 轮廓数组, 轮廓索引(-1为所有), 颜色(红色), 线宽
        cv2.drawContours(shape_img, [hull], -1, (0, 0, 255), 2)
        
        # 如果需要更平滑的效果，可以画出关键点（可选）
        # for point in hull:
        #     cv2.circle(shape_img, tuple(point[0]), 3, (0, 255, 0), -1)

    # 输出结果与文件保存
    print("--- 处理完成 ---")
    print(f"检测汉字总数: {len(final_candidates)}")
    print(f"🔴 长形字: {count_long}, 🟢 方形字: {count_square}, 🔵 扁形字: {count_flat}")

    # 文件名处理
    dir_name = os.path.dirname(image_path)
    base_name, ext_name = os.path.splitext(os.path.basename(image_path))
    
    # 保存分析框图
    output_analyzer = os.path.join(dir_name, f"{base_name}_analyzer{ext_name}")
    cv2.imwrite(output_analyzer, analyzer_img)
    print(f"✅ 框型图已保存: {output_analyzer}")

    # 保存形状图 (新需求)
    output_shape = os.path.join(dir_name, f"{base_name}_shape{ext_name}")
    cv2.imwrite(output_shape, shape_img)
    print(f"✅ 形状图已保存: {output_shape}")
    
    # 显示结果 (显示形状图预览)
    plt.figure(figsize=(12, 18))
    plt.title("Convex Hull Shape Analysis")
    plt.imshow(cv2.cvtColor(shape_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # 请在此处替换你的文件路径
    process_calligraphy('./3.jpg')
