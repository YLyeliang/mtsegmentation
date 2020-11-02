def bbox_equa(A, B):
    """
    compare two bboxes whether they are the same
    :return:
    """
    assert A is not None and B is not None
    Axmin, Aymin, Axmax, Aymax, Ascore = A
    Bxmin, Bymin, Bxmax, Bymax, Bscore = B

    if Axmin == Bxmin and Aymin == Bymin:
        if Axmax == Bxmax and Aymax == Bymax:
            return True
    else:
        return False


def iou(A, B):
    assert A is not None and B is not None
    Axmin, Aymin, Axmax, Aymax, Ascore = A
    Bxmin, Bymin, Bxmax, Bymax, Bscore = B

    inter_xmin = max(Axmin, Bxmin)
    inter_ymin = max(Aymin, Bymin)

    inter_xmax = min(Axmax, Bxmax)
    inter_ymax = min(Aymax, Bymax)

    area_A = (Axmax - Axmin) * (Aymax - Aymin)
    area_B = (Bxmax - Bxmin) * (Bymax - Bymin)
    area_intersect = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    IoU = area_intersect / (area_A + area_B - area_intersect)
    return IoU


def merge_bbox(A, B, diff_cond=30):
    assert A is not None and B is not None
    # 将两个横向相邻的bbox进行拼接
    # 如果两个bboxes的y轴坐标相差不大，且相邻的距离不大，
    # 或者 一个bbox 包含另一个bbox，则相邻
    Axmin, Aymin, Axmax, Aymax, Ascore = A
    Bxmin, Bymin, Bxmax, Bymax, Bscore = B

    ymin_diff = abs(Aymin - Bymin)
    ymax_diff = abs(Aymax - Bymax)

    x1_diff = abs(Axmin - Bxmax)
    x2_diff = abs(Axmax - Bxmin)
    x3_diff = abs(Axmin - Bxmin)
    x4_diff = abs(Axmax - Bxmax)
    x_diff = min(x1_diff, x2_diff,x3_diff,x4_diff)
    score = max(Ascore, Bscore)

    if ymin_diff < diff_cond and ymax_diff < diff_cond and x_diff < diff_cond:
        xmax_new = Axmax if Axmin >= Bxmax else Bxmax
        xmin_new = Bxmin if Axmin >= Bxmax else Axmin
        ymin_new = min(Aymin, Bymin)
        ymax_new = max(Aymax, Bymax)
        box_new = [xmin_new, ymin_new, xmax_new, ymax_new, score]
        return box_new
    return None


def bbox_area(A):
    """
    calculate area of a bbox.
    :param A:
    :return:
    """
    xmin, ymin, xmax, ymax, score = A
    width = xmax - xmin
    height = ymax - ymin
    return width * height
