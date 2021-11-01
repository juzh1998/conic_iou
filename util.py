import numpy as np
import math

def para2mat(conic):
    c_x, c_y, a, b, th = conic[0], conic[1], conic[2], conic[3], conic[4]
    cos_val, sin_val = math.cos(th), math.sin(th)
    transform_mat = np.array([[cos_val, -1 * sin_val, 0], [sin_val, cos_val, 0],
                              [-1 * c_x * cos_val - c_y * sin_val, c_x * sin_val - c_y * cos_val, 1]])
    base_A_mat = np.array(
        [[b * b, 0, 0], [0, a * a, 0], [0, 0, -1 * a * a * b * b]])
    transformed_A_mat = (transform_mat.dot(base_A_mat)).dot(transform_mat.T)
    return transformed_A_mat


def conic_point(conic_para):
    # 参数：椭圆中心点坐标，ab轴长度，旋转角度参数
    x, y, a, b, theta = conic_para[0], conic_para[1], conic_para[2], conic_para[3], conic_para[4]
    x_1, y_1 = a * math.cos(theta) + x, y + a * math.sin(theta)
    x_2, y_2 = b * math.cos(theta + math.pi / 2) + \
        x, y + b * math.sin(theta + math.pi / 2)
    x_3, y_3 = a * math.cos(theta + math.pi) + x, y + \
        a * math.sin(theta + math.pi)
    x_4, y_4 = b * math.cos(theta - math.pi / 2) + \
        x, y + b * math.sin(theta - math.pi / 2)
    point = np.array([[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4]])
    return point


def convex_areas(c):
    c_1 = (c[0, :]).reshape(1, 2)
    c = np.append(c, c_1, axis=0)
    res = 0.0
    for i in range(c.shape[0] - 1):
        res = res + c[i, 0] * c[i + 1, 1] - c[i, 1] * c[i + 1, 0]
    res = res / 2
    return res


def sort_point(c):
    mean_location = np.mean(c, axis=0)
    mean_location = mean_location.reshape((1, 2))
    c_new = c - mean_location
    sort_list = []

    for i in range(c.shape[0]):
        sort_list.append(math.atan2(c_new[i, 1], c_new[i, 0]))
    vals = np.array(sort_list)
    index = np.argsort(vals)
    c_sort = c[index, :]
    return c_sort


def point_at_conic(a_point, conic_para, flag=0):
    x, y, a, b, theta = conic_para[0], conic_para[1], conic_para[2], conic_para[3], conic_para[4]
    c = math.sqrt(a**2 - b**2)
    c1_x, c1_y = x + c * math.cos(theta), y + math.sin(theta) * c
    c2_x, c2_y = x + c * math.cos(theta + math.pi), y + \
        math.sin(theta + math.pi) * c

    if flag == 0:  # 判断点是否在椭圆上
        d1 = math.sqrt((a_point[0] - c1_x)**2 + (a_point[1] - c1_y)**2)
        d2 = math.sqrt((a_point[0] - c2_x)**2 + (a_point[1] - c2_y)**2)
        if abs(d1 + d2 - 2 * a) < 0.0001:
            return True
        else:
            return False
    else:  # 判断点是否在椭圆里面
        point = []
        for i in range(a_point.shape[0]):
            temp_point = a_point[i, :]
            d1 = math.sqrt((temp_point[0] - c1_x) **
                           2 + (temp_point[1] - c1_y) ** 2)
            d2 = math.sqrt((temp_point[0] - c2_x) **
                           2 + (temp_point[1] - c2_y) ** 2)
            if (d1 + d2) < 2 * a:
                point.append([temp_point[0], temp_point[1]])
        return point


def contact_point_and_line(conic_para, c, flag=0, *conic_para_2):
    # k旋转后切线斜率，conic_para椭圆参数,c：已知的两个交点
    # 算法思路：将切线和椭圆旋转平移到到原点，然后计算出两个切点，再将切点旋转平移回到现在坐标系下
    # 并利用c中两个交点位置确定唯一的切点
    # 根据切点坐标计算出切线y=kx+m中的m
    x, y, a, b, theta = conic_para[0], conic_para[1], conic_para[2], conic_para[3], conic_para[4]
    cos_val, sin_val = math.cos(theta), math.sin(theta)
    c1_x, c1_y, c2_x, c2_y = c[0, 0], c[0, 1], c[1, 0], c[1, 1]

    # 将交点平移旋转回原点坐标系
    c1_x_pre, c1_y_pre = (c1_x - x) * cos_val + (c1_y - y) * \
        sin_val, -1 * (c1_x - x) * sin_val + (c1_y - y) * cos_val
    c2_x_pre, c2_y_pre = (c2_x - x) * cos_val + (c2_y - y) * \
        sin_val, -1 * (c2_x - x) * sin_val + (c2_y - y) * cos_val

    if c1_x_pre - c2_x_pre != 0:
        k_1 = (c2_y_pre - c1_y_pre) / (c2_x_pre - c1_x_pre)

        temp_func = (a * k_1)**2 + b**2
        m_1, m_2 = math.sqrt(temp_func), -1 * math.sqrt(temp_func)

        x_1, x_2 = -1 * (a * a * k_1 * m_1) / (a * a * k_1 * k_1 + b * b), \
            -1 * (a * a * k_1 * m_2) / (a * a * k_1 * k_1 + b * b)
        y_1, y_2 = k_1 * x_1 + m_1, k_1 * x_2 + m_2
    else:
        x_1, y_1 = a, 0
        x_2, y_2 = -1 * a, 0

    x_1_af = x_1 * cos_val - y_1 * sin_val + x
    y_1_af = x_1 * sin_val + y_1 * cos_val + y

    x_2_af = x_2 * cos_val - y_2 * sin_val + x
    y_2_af = x_2 * sin_val + y_2 * cos_val + y
    if flag == 0:
        conic_para_2=conic_para_2[0]
        point = np.array([[x_1_af, y_1_af], [x_2_af, y_2_af]])
        point_contact = point_at_conic(point, conic_para_2, 1)
        if len(point_contact) == 2:

            d1_1 = math.sqrt(
                ((point_contact[0])[0] - c1_x) ** 2 + ((point_contact[0])[1] - c1_y) ** 2)
            d1_2 = math.sqrt(
                ((point_contact[0])[0] - c2_x) ** 2 + ((point_contact[0])[1] - c2_y) ** 2)
            d2_1 = math.sqrt(
                ((point_contact[1])[0] - c1_x) ** 2 + ((point_contact[1])[1] - c1_y) ** 2)
            d2_2 = math.sqrt(
                ((point_contact[1])[0] - c2_x) ** 2 + ((point_contact[1])[1] - c2_y) ** 2)
            point=[]
            if (d1_1 + d1_2) < (d2_2 + d2_1):

                point.append(point_contact[0])
            else:
                point.append(point_contact[1])

        return point_contact
    else:
        distance_1 = (abs(x_1_af - c1_x)) ** 2 + (abs(y_1_af - c1_y)) ** 2
        distance_2 = (abs(x_2_af - c2_x)) ** 2 + (abs(y_2_af - c2_y)) ** 2
        if distance_1 < distance_2:
            point = np.array([x_1_af, y_1_af])
        else:
            point = np.array([x_2_af, y_2_af])
        return point


def cacu_contact_M(c1, c2, A, B, k, C):
    if c1 in C and c2 in C:  # 如果两个都是交点，则使用之前的方法求切点
        c = np.append(c1.reshape(1, -1), c2.reshape(1, -1), axis=0)
        point1 = contact_point_and_line(A, c, 0, B)
        point2 = contact_point_and_line(B, c, 0, A)
        point = point1 + point2
        if len(point) == 2:

            d1_1 = math.sqrt(((point[0])[0] - c1[0]) **
                             2 + ((point[0])[1] - c1[1]) ** 2)
            d1_2 = math.sqrt(((point[0])[0] - c2[0]) **
                             2 + ((point[0])[1] - c2[1]) ** 2)
            d2_1 = math.sqrt(((point[1])[0] - c1[0]) **
                             2 + ((point[1])[1] - c1[1]) ** 2)
            d2_2 = math.sqrt(((point[1])[0] - c2[0]) **
                             2 + ((point[1])[1] - c2[1]) ** 2)
            if (d1_1 + d1_2) < (d2_2 + d2_1):
                point = point[0]
            else:
                point = point[1]
        point = np.array(point)

    else:  # 如果是交点和端点或者切点，则使用现在的方法
        point_colleciton = np.array([[c1[0], c1[1]], [c2[0], c2[1]]])
        if point_at_conic(c1, A) and point_at_conic(c2, A):

            point = contact_point_and_line(A, point_colleciton, 1)
        else:
            point = contact_point_and_line(
                B, point_colleciton, 1)  # 在椭圆二上距离两个交点最近的切点

    if k is None:
        M = None
    else:
        M = point[1] - k * point[0]
    # 返回切点以及截距
    return point, M


def sub_caculate_inter_area_2(c, A, B, C):
    # 利用iou外接四边形近似iou
    # 计算iou外接四边形/三边形切线
    # C 是原有的交点
    c = sort_point(c)
    c_1 = (c[0, :]).reshape(1, 2)
    c = np.append(c, c_1, axis=0)
    k, contact_m, contact_point = [], [], []
    for i in range(len(c) - 1):
        c1, c2 = c[i, :], c[i + 1, :]

        if abs(c1[0] - c2[0]) <= 0.0005:
            temp_k = None
        else:
            temp_k = (c1[1] - c2[1]) / (c1[0] - c2[0])
        k.append(temp_k)
        temp_contact_point, temp_m = cacu_contact_M(c1, c2, A, B, temp_k, C)
        contact_m.append(temp_m)
        contact_point.append(temp_contact_point)

    # 计算iou外接四边形/三边形交点
    k.append(k[0])
    contact_m.append(contact_m[0])
    contact_point.append(contact_point[0])

    for i in range(len(k) - 1):
        if (k[i] is None) or (k[i + 1] is None):
            # 两条切线中有一条没有斜率
            if k[i] is None:
                x = (contact_point[i])[0]
                y = k[i + 1] * x + contact_m[i + 1]
            else:
                x = (contact_point[i + 1])[0]
                y = k[i] * x + contact_m[i]
        else:
            x = (contact_m[i + 1] - contact_m[i]) / (k[i] - k[i + 1])
            y = x * k[i] + contact_m[i]
        temp_point = np.array([x, y])
        if i == 0:
            joint_point = temp_point.reshape(1, -1)
        else:
            temp_point = temp_point.reshape(1, -1)
            joint_point = np.append(joint_point, temp_point, axis=0)

    # 计算外接四边形面积
    I = convex_areas(joint_point)
    return I


def caculate_inter_area(a, b, c):
    I = 0
    c = np.unique(c, axis=0)

    # 使用外切四边形计算iou面积
    if c.shape[0] == 3 or c.shape[0] == 4:
        I = sub_caculate_inter_area_2(c, a, b, c)
    elif c.shape[0] == 2:
        # 将这个部分用一个四边形近似
        A_contact_point = contact_point_and_line(a, c, 0, b)
        B_contact_point = contact_point_and_line(b, c, 0, a)
        # print(A_contact_point)
        # print(B_contact_point)
        c_new = np.append(c, A_contact_point, axis=0)
        c_new = np.append(c_new, B_contact_point, axis=0)
        I = sub_caculate_inter_area_2(c_new, a, b, c)

    else:
        I = 0

    return I


def caculate_iou(a, b, c, flag=0):
    if flag == 1:
        area1 = 4 * a[2] * a[3]
        area2 = 4 * b[2] * b[3]
    else:
        area1 = math.pi * a[2] * a[3]
        area2 = math.pi * b[2] * b[3]
    I = caculate_inter_area(a, b, c)
    # print(I)
    # print(area2+area1-I)
    conic_IOU = I / (area1 + area2 - I)
    # print(conic_IOU)
    return conic_IOU
