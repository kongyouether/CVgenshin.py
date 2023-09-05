#步骤就是1先检测四个角点进行透视变换矫正2检测蓝色方块位置，用于进行旋转矫正3直接对矫正完成后的图像进行霍夫圆检测4对得到的像素坐标点
#转化到需要的参考系中，注释和代码还没详细整理，先这么写着，后面再把视频流补出来。20230902

#导
import cv2
import numpy as np
from PyQt5.QtGui import QImage
global result_image
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
#函数定义
def test_main(image): 

    # global processed_image
    # global all_circle_coordinates
    # global result_image
        # 辅助函数，用于将 QImage 转换为 NumPy 数组
        
        # 辅助函数，用于将 NumPy 数组转换为 QImage
    def cv2_image_to_qimage(cv2_image):
        height, width, channel = cv2_image.shape
        bytes_per_line = 3 * width
        return QImage(cv2_image.data, width, height, bytes_per_line, QImage.Format_BGR888)


    def qimage_to_ndarray(qimage):
        width = qimage.width()
        height = qimage.height()
        bytes_per_line = qimage.bytesPerLine()
        image_data = qimage.bits().asstring()

        return np.frombuffer(image_data, dtype=np.uint8).reshape(height, width, 4)

    def cvimage_to_qimage(cvimage):
        height, width, channel = cvimage.shape
        bytes_per_line = 3 * width
        # 将 BGR 转换为 RGB
        cvimage_rgb = cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB)
        qimage = QImage(cvimage_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return qimage
    
    def qimage_to_cvimage(qimage):
        width = qimage.width()
        height = qimage.height()
        bytes_per_line = qimage.bytesPerLine()
        data_ptr = qimage.bits()

        # 确保图像具有3通道
        if qimage.format() == QImage.Format_RGB888:
            image = np.ndarray((height, width, 3), dtype=np.uint8, buffer=data_ptr, strides=(bytes_per_line, 3, 1))
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            # 如果图像不是3通道的RGB格式，可以进行适当的处理
            # 这里你可以添加其他处理逻辑
            return None



    def rotate_image(image, num_rotations):
        # 旋转图像
        for _ in range(num_rotations):
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        
        return image

    def determine_quadrant_pixel_coordinates(point, image_width, image_height):#检测图像中的蓝色区域
        x, y = point
        half_width = image_width // 2
        half_height = image_height // 2
        rotate_num=0
        if x < half_width and y < half_height:
            rotate_num=0
            return "左上",rotate_num
        elif x >= half_width and y < half_height:
            rotate_num=3
            return "右上",rotate_num
        elif x >= half_width and y >= half_height:
            rotate_num=2
            return "右下",rotate_num
        elif x < half_width and y >= half_height:
            rotate_num=1
            return "左下",rotate_num
        else:
            rotate_num=0
            return "未知",rotate_num
        
    def symmetric_point(point1, point2):#找到点对称点
        x1, y1 = point1
        x2, y2 = point2

        # 计算对称点的坐标
        x3 = 2 * x2 - x1
        y3 = 2 * y2 - y1

        return (x3, y3)

    def find_fourth_point_qr_code(points):
        if len(points) != 3:
            raise ValueError("Input 'points' must contain exactly three points.")

        # 将输入的点坐标转换为NumPy数组
        A_first = find_largest_angle_point(points)
        points_copy=points.copy()
        points_copy.remove(A_first)
        points_copy.append(A_first)
        #把定位点放到最后一个的位置

        print("points_copy",points_copy)



        #自己备份自己

        #把定位点放到列表最后一个

        input_points = np.array(points_copy, dtype=np.float32)

        # 计算第四个点的坐标，假设三个输入点分别为A、B、C

        C, B, A = input_points
        print("C",C)
        print("B",B)
        print("A",A)
        #A, C, B = input_points

        # 计算D点，假设D点在AB向量上与C点相同距离
        AB = B - A
        D = C + AB
        D_point = (int(D[0]), int(D[1]))
        #D_point=tuple(fourth_point)
        #转化为元组。可以用circle显示

        return D_point

    def find_largest_angle_point(points):#找到直角点
        if len(points) != 3:
            raise ValueError("Input 'points' must contain exactly three points.")

        # 计算各个边的长度
        #所对边即该角

        a = np.linalg.norm(np.array(points[1]) - np.array(points[2]))
        point_a=points[0]
        b = np.linalg.norm(np.array(points[0]) - np.array(points[2]))
        point_b=points[1]
        c = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
        point_c=points[2]

        print("a",a)
        print("b",b)
        print("c",c)

        # 计算各个角的余弦值
        cos_A = (b**2 + c**2 - a**2) / (2 * b * c)
        cos_B = (a**2 + c**2 - b**2) / (2 * a * c)
        cos_C = (a**2 + b**2 - c**2) / (2 * a * b)

        # 找到最小余弦值对应的点
        if cos_A < cos_B and cos_A < cos_C:
            print("Amax")
            return point_a
        elif cos_B < cos_A and cos_B < cos_C:
            print("Bmax")
            return point_b
        else:
            print("Cmax")
            return point_c
        

    def add_gaussian_noise(image, mean, std_dev):
        height, width, channels = image.shape
        noise = np.random.normal(mean, std_dev, (height, width, channels))
        noisy_image = cv2.add(image, noise.astype(np.uint8))
        return noisy_image

    def transform_coordinates(pixel_coords):#将像素点坐标映射到笛卡尔坐标
        new_coords = []
        for x, y in pixel_coords:
            new_x = min(max((x // 50) + 1, 1), 8)
            newy = min(max((y // 50) + 1, 1), 8)
            new_y = 9 - newy
            new_coords.append((new_x, new_y))
        return new_coords

    # def perspective_transform_three(image, points):
    #     if len(points) != 3:
    #         raise ValueError("Input 'points' must contain exactly three points.")
        
    #     # 将输入的点坐标转换为NumPy数组
    #     input_points = np.array(points, dtype=np.float32)

    #     # 计算第四个点的坐标，假设三个输入点分别为A、B、C
    #     A, B, C = input_points
    #     D = C + (B - A)  # D是C到B的向量加上A的坐标

    #     # 定义输出图像上的对应点，这些点将映射到新的位置
    #     output_points = np.array([[0, 0], [image.shape[1], 0], [0, image.shape[0]], [D[0], D[1]]], dtype=np.float32)

    #     # 计算透视变换矩阵
    #     matrix = cv2.getPerspectiveTransform(input_points, output_points)

    #     # 执行透视变换
    #     transformed_image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

    #     return transformed_image

    def detect_blue_center(image):#检测蓝色区域中心

        if image is None:
            raise ValueError("无法读取图片")

        # 将图像从BGR颜色空间转换为HSV颜色空间
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 定义蓝色的HSV范围（可以根据需要进行调整）
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # 创建蓝色掩码
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

        # 查找蓝色区域的轮廓
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            raise ValueError("未检测到蓝色区域")

        # 计算蓝色区域的中心
        blue_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(blue_contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            center_x, center_y = 0, 0

        # 在图像上标记中心
        #cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), -1)

        #cv2.putText(image, f"({center_x}, {center_y})", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 返回中心坐标
        return (center_x, center_y)

    def perspective_transform_rgb(input_points, frame):#对RBG通道的图像进行处理
        # 定义输出图像的四个角点（左上、右上、左下、右下）
        
        output_width = 550#其实550还可以放大,还会影响霍夫圆的检测
        output_height = 550
        output_points = np.array(
            [[0, 0], [output_width - 1, 0], [0, output_height - 1], [output_width - 1, output_height - 1]],
            dtype=np.float32)

        # 将输入点和输出点转换为适合OpenCV函数的格式
        input_points = np.array(input_points, dtype=np.float32)
        output_points = np.array(output_points, dtype=np.float32)

        # 获取输入图像的通道数
        num_channels = frame.shape[2] if len(frame.shape) == 3 else 1

        # 计算透视变换矩阵
        transform_matrix = cv2.getPerspectiveTransform(input_points, output_points)

        # 进行透视变换
        if num_channels == 1:  # 灰度图像
            output_frame = cv2.warpPerspective(frame, transform_matrix, (output_width, output_height), flags=cv2.INTER_LINEAR)
        elif num_channels == 3:  # RGB图像
            output_frame = cv2.warpPerspective(frame, transform_matrix, (output_width, output_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        else:
            raise ValueError("不支持的通道数")

        return output_frame

    def perspective_transform(input_points, frame):
        # 定义输出图像的四个角点（左上、右上、左下、右下）
        output_width = 550
        output_height = 550
        output_points = np.array(
            [[0, 0], [output_width - 1, 0], [0, output_height - 1], [output_width - 1, output_height - 1]],
            dtype=np.float32)

        # 将输入点和输出点转换为适合OpenCV函数的格式
        input_points = np.array(input_points, dtype=np.float32)
        output_points = np.array(output_points, dtype=np.float32)

        # 计算透视变换矩阵
        transform_matrix = cv2.getPerspectiveTransform(input_points, output_points)

        # 获取输入图像的高度和宽度
        height, width = frame.shape[:2]

        # 进行透视变换
        output_frame = cv2.warpPerspective(frame, transform_matrix, (output_width, output_height))

        return output_frame

    def detect_circles(image):#检测霍夫圆

        # 转换图像类型为彩色（BGR格式）
        image1 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 使用Hough变换检测圆
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=17, minRadius=5,
                                    maxRadius=15)
        #将参数param2调到15是比较合适的

        circle_centers = []

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            for (x, y, r) in circles:
                # 排除直径大于100像素的圆
                if r < 100:
                    # 在圆周上绘制圆
                    cv2.circle(image1, (x, y), r, (0, 255, 0), 4)
                    # 绘制圆心
                    cv2.circle(image1, (x, y), 2, (0, 0, 255), 3)
                    # 添加圆心坐标到列表中
                    
                    #对坐标点进行映射
                    (x1, y1)=(x-75, y-75)
                    x2=min(max((x1 // 50) + 1, 1), 8)
                    y22= min(max((y1 // 50) + 1, 1), 8)
                    y2 = 9 - y22
                    x3=x2-1
                    y3=8-y2
                    cv2.putText(image1, f"({x3}, {y3})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    # 绘制坐标信息
                    circle_centers.append((x3, y3))
                    # cv2.putText(image1, f"({x}, {y})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


        return image1, circle_centers

    def determine_relative_positions(points):#排列四个角点
        # 根据横坐标排序
        sorted_by_x = sorted(points, key=lambda p: p[0])

        # 左边的点
        left_points = sorted_by_x[:2]

        # 右边的点
        right_points = sorted_by_x[2:]

        # 根据纵坐标排序
        sorted_by_y_left = sorted(left_points, key=lambda p: p[1])
        sorted_by_y_right = sorted(right_points, key=lambda p: p[1])

        # 左上方的点
        left_top = sorted_by_y_left[0]

        # 左下方的点

        left_bottom = sorted_by_y_left[1]
        # 右下方的点
        right_bottom = sorted_by_y_right[1]

        # 右上方的点
        right_top = sorted_by_y_right[0]

        return left_top, left_bottom, right_bottom, right_top


    def otsu_threshold(image):#大津法二值化
        # 读取图像
        
        # 将图像转换为灰度图像image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用大津法计算最佳阈值
        _, threshold_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = 0 # 提取阈值并转换为整数类型
        #所以这个最佳阈值怎么搞
        
        # 应用阈值进行二值化
        _, binary_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        return binary_image

    def shift_point(point):#将点往下移20个像素
        global result_image
        new_x = point[0]
        new_y = point[1]+15
        return (new_x, new_y)

   #主程序
    #cangbaotu=image
        # 将 QImage 转换为 BGR 格式的 NumPy 数组

    #image_np = qimage_to_ndarray(image)
    cangbaotu = image
    #cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #cangbaotu=cv2.imread('G:/vscodetest/perspective4.png')
    #读取藏宝图
    
    gray = cv2.cvtColor(cangbaotu, cv2.COLOR_BGR2GRAY)

    if cangbaotu is  None:
        print("未正常读取")
    #读取失败

    if cangbaotu is not None:
    #读取成功

        #gray = cv2.cvtColor(cangbaotu, cv2.COLOR_BGR2GRAY)
        #转化为灰度图

    # _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        #转化为二值化图

        #cv2.imshow("binary",binary)

        #cv2.imshow("cangbaotu",cangbaotu)

        zhongzhilvbo=cv2.medianBlur(cangbaotu,3)
        #进行中值滤波

        #cv2.imshow("zhongzhilvbo",zhongzhilvbo)

        gaosilvbo=cv2.GaussianBlur(cangbaotu,(5,5),sigmaX=1)
        #进行高斯滤波

        #cv2.imshow("gaosilvbo",gaosilvbo)

        binary_image1 = otsu_threshold(cangbaotu)

        _, binary_image2 = cv2.threshold(zhongzhilvbo, 127, 255, cv2.THRESH_BINARY)
        #运行普通方法来进行二值化图像

        # 显示原始图像和二值化图像
        #cv2.imshow("Original Image",cangbaotu)

        #cv2.imshow("Binary Image1", binary_image1)
        #cv2.imshow("Binary Image2", binary_image2)



        edges = cv2.Canny(binary_image1, 50, 150)

        edges = cv2.Canny(gray, 50, 150)
        #不进行大津法，直接用灰度图进行边缘检测

        #进行边缘检测

        #cv2.imshow("edges",edges)

        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #对轮廓进行检测，RETR_EXTERNAL是外轮廓，

        squares = []

        # 遍历所有轮廓
        for contour in contours:
            # 近似多边形轮廓
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * peri, True)

            # 如果近似轮廓具有4个顶点且是凸包，则认为它是一个正方形

            if len(approx) == 4 and cv2.isContourConvex(approx):
                # 计算中点坐标
                M = cv2.moments(approx)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # 计算正方形大小
                size = cv2.contourArea(approx)

                # 只保留像素大小在400到1500之间的正方形
                if 400 < size < 5000:
                    squares.append((approx, (cX, cY), size))
                    #得到一个正方形列表
                #print("squares",squares)    
                if 5000 < size :
                    cv2.drawContours(cangbaotu, [approx], -1, (0, 255, 0), 2)
                    #cv2.circle(cangbaotu, (cX, cY), 10, (100, 0, 255), 5)
                    point_center=(cX, cY)
                    #cv2.putText(cangbaotu, f"Size: {size}", point_center, cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            # (0, 255, 0),
                            # 1)

                for square in squares:

                    approx = square[0]
                    cv2.drawContours(cangbaotu, [approx], -1, (0, 255, 0), 2)
                    cv2.circle(cangbaotu, square[1], 3, (0, 0, 255), -1)
                    # cv2.putText(cangbaotu, f"Size: {int(square[2])}", square[1], cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    #             (0, 255, 0),
                    #             1)
                    
                detected_circles = []
                circle_count = 1
                

            center_points = [square[1] for square in squares]

        print("难道他真有四个?")

        if len(center_points)==3:#定位点为三个时
            
            print("三个是吧")
            print("center_points",center_points)

            largest_angle_point = find_largest_angle_point(center_points)
            #确定定位点中的直角点
            
            cv2.circle(cangbaotu, largest_angle_point, 10, (255, 0, 255), -1)
            cv2.putText(cangbaotu, " dingweidian", largest_angle_point, cv2.FONT_HERSHEY_SIMPLEX, 1,
            (255, 0, 255),
            2)
            #画出定位点中的直角点

            print("最大角对应的点坐标：", largest_angle_point)  


            fourth_point=symmetric_point(largest_angle_point,point_center)     
            #得到第四个点的简单方法。
            
            #fourth_point = find_fourth_point_qr_code(center_points)
            
            print("fourth_point data type:", type(fourth_point))
            cv2.circle(cangbaotu, fourth_point, 10, (255, 0, 255), -1)
            #预测第四个点

            print("拟合的第四个点坐标：", fourth_point)    
            #会有负数的问题
                        
        # output_image_three =perspective_transform_three(binary_image1,center_points )
            #cv2.imshow('Output Image', output_image_three)
            #显示检测霍夫圆后的藏宝图
            #cv2.imshow("result_image", result_image)

            #cv2.imshow("xianshixinxi", cangbaotu)


        if len(center_points)==4:#定位点为四个时
            print("坏了，他真有四个")


            print("center_points",center_points)
                    #找到四个定位点的中心

            top_left, bottom_left, bottom_right, top_right = determine_relative_positions(center_points)
            #对藏宝图的四个定位点进行检测
            
            #显示坐标
            cv2.putText(cangbaotu, f"Top Left: ", top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                            (255, 0, 0),
                                            1)
            #将Top Left用f"Top Left: {top_left}"变成一个字符串来显示
            cv2.putText(cangbaotu, f"bottom_left: {bottom_left}", bottom_left, cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                            (255, 0, 0),
                                            1)
            cv2.putText(cangbaotu, f"top_right: {top_right}", top_right, cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                            (255, 0, 0),
                                            1)
            cv2.putText(cangbaotu, f"bottom_righ: {bottom_right}", bottom_right, cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                            (255, 0, 0),
                                            1)        
            shift_top_left=shift_point(top_left)  
            #在下一行显示坐标值   
            
            cv2.putText(cangbaotu, f" {top_left}", shift_top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (255, 0, 0),
                            1)
            
            print("Top Left:", top_left)
            print("Bottom Left:", bottom_left)
            print("Top Right:", top_right)
            print("Bottom Right:", bottom_right)
            print("------------------------")
            #输出四个定位点的位置

            input_points = [[top_left], [top_right], [bottom_left], [bottom_right]]

        
            output_cangbaotu = perspective_transform(input_points, gray)  
            #对灰度化后图片进行透视变换

            output_cangbaotu_rbg = perspective_transform_rgb(input_points, cangbaotu) 

            center_coordinates = detect_blue_center(output_cangbaotu_rbg)  # 替换为你的输入图像文件路径
            
            print("蓝色区域的中心坐标:", center_coordinates)

            #cv2.imshow("output_cangbaotu_rbg", output_cangbaotu_rbg)

            quadrant,rotate_num = determine_quadrant_pixel_coordinates(center_coordinates,550,550)
            #确定蓝色点的位置在哪个区域

            print("\n")
            print("点位于", quadrant)

            rotated_image = rotate_image(output_cangbaotu_rbg,rotate_num)

            #跟据蓝色点位置旋转图像

            #cv2.imshow("rotated_image", rotated_image)
            #显示旋转矫正后的图像

            gray_rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
            #将旋转矫正后的图像转化为灰度图

            result_image, centers = detect_circles(gray_rotated_image)
            #霍夫圆检测藏宝图中的圆

            for center in centers:
                detected_circles.append(center)
                print(f"圆{circle_count}的圆心坐标:", center)
                circle_count += 1
            all_circle_coordinates = detected_circles[:8]  # 获取前8个圆的坐标
            pixel_coordinates = all_circle_coordinates
            updated_coordinates = [[x - 75, y - 75] for x, y in pixel_coordinates]
            #通过映射关系将像素坐标圆点映射到新的坐标中

            print("所有圆的坐标：", all_circle_coordinates)
            #在外面在映射一次，方便说明

            print("新坐标：", centers)
            #直接在用函数里面的数据
            
            
            # if centers is not None:
            #     print("centers",centers)
#换一种吧
            # contours, _ = cv2.findContours(output_cangbaotu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # #对轮廓进行检测，RETR_EXTERNAL是外轮廓，

            # squares = []

            # # 遍历所有轮廓
            # for contour in contours:
            #     # 近似多边形轮廓
            #     peri = cv2.arcLength(contour, True)
            #     approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            #     # 如果近似轮廓具有4个顶点且是凸包，则认为它是一个正方形

            #     if len(approx) == 4 and cv2.isContourConvex(approx):
            #         # 计算中点坐标
            #         M = cv2.moments(approx)
            #         cX = int(M["m10"] / M["m00"])
            #         cY = int(M["m01"] / M["m00"])

            #         # 计算正方形大小
            #         size = cv2.contourArea(approx)

            #         # 只保留像素大小在400到1500之间的正方形
    
            #         if 5000 < size :
            #             cv2.drawContours(result_image, [approx], -1, (0, 255, 0), 2)
            #             cv2.circle(result_image, square[1], 3, (0, 0, 255), -1)
            #             cv2.putText(result_image, f"Size: {size}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
            #                         (0, 255, 0),
            #                         1)                

            #显示检测霍夫圆后的藏宝图
            #cv2.imshow("result_image", result_image)
            #cv2.imshow("xianshixinxi", cangbaotu)
        else:
            print("如有")

        #按下Q结束显示
        # while True:
        #     key= cv2.waitKey(1)
        #     if key==ord('q')or key ==ord('Q'):
        #         cv2.destroyAllWindows()
        #         break
        print("已经结束哩")
    # 最后，将处理后的图像转换回 QImage 格式
    # if len(center_points)!=3 and len(center_points)!=4 :
    #     result_image=Image.open("zhengque.png")
    #     processed_image=Image.open("zhengque.png")
    #     all_circle_coordinates=["哼！","(◍•ᴗ•◍)"]

    if len(center_points)==4 :
        processed_image = cvimage_to_qimage(result_image)
    else:
        result_image = cv2.imread("zhengque.png")
        #result_image=Image.open("zhengque.png")
        processed_image=cv2.imread("zhengque.png")
        #cv2.imshow("processed_image",processed_image)
        all_circle_coordinates=["快去选择正确图片捏~""哼！(◍•ᴗ•◍)"]
        # 确保图像为RGB格式
        result_image = cv2_image_to_qimage(result_image)
        processed_image = cv2_image_to_qimage(processed_image)
        #cv2.imshow("processed_image",processed_image)

    return processed_image,all_circle_coordinates,result_image
    #return result_image
