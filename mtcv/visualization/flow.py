from graphviz import Digraph
from dropblock import DropBlock2D


def flow_chart():
    dot = Digraph(comment='The test Table', format='png')
    dot.node('a', '原始实例')
    dot.node('b', 'TB抠图')
    dot.node('c', 'TB图像增强')
    dot.node('d', '训练样本生成')
    dot.node('e', '模型训练')
    dot.node('f', '实际测试')
    dot.edges(['ab', 'bc', 'cd', 'de', 'ef'])
    dot.view()


# flow_chart()

def framework_chart():
    dot = Digraph(comment='tbdetection_framework', format='png')
    dot.node('a', '检测框架')
    dot.node('b', '数据部分')
    dot.node('c', '检测部分')
    dot.node('d', '特效库')
    dot.node('e', '融合方法')
    dot.node('f', '批量生成api')
    dot.node('g', '训练格式转换')
    dot.node('p', 'dataset')
    dot.node('q','mosaic,grid, et al.')
    dot.node('h', 'backbone')
    dot.node('i', 'neck')
    dot.node('j', 'head')
    dot.node('k', 'detector')
    dot.node('l', 'res,csp')
    dot.node('m', 'fpn,panet')
    dot.node('n', 'refinedet,retina')
    dot.node('o', 'refinedet,retina')
    dot.edges(['ab', 'ac', 'bd', 'be', 'bf', 'bg', 'cp', 'ch', 'ci', 'cj', 'ck', 'hl', 'im', 'jn', 'ko','pq'])
    dot.view()


def autotrain_flow():
    dot = Digraph(comment='autotrain_framework', format='png')
    dot.node('A', '构建配置文件')
    dot.node('B', '解析配置文件')
    dot.node('C', 'cfg:data,cfg:train')
    dot.node('D', '数据生成，保留输出文件目录')
    dot.node('E', '更新cfg:train中的数据集目录')
    dot.node('F', '利用更新后的数据集进行训练')
    dot.node('G', '训练完成后model.eval()对测试集进行测试')
    dot.node('H', '对权重文件排序，取后3个epoch进行测试，取最佳模型')
    dot.node('I', '保留最佳权重文件')
    dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG', 'GH', 'HI'])
    dot.view()


framework_chart()
# autotrain_flow()
