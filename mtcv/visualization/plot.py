import matplotlib.pyplot as plt
from skimage.util import montage


def plot():
    a = [43, 22, 18, 17.9, 17.8, 17.877, 17.966, 17.677, 17.878, 17.666, 17.555, 17.877]
    print(len(a))
    b = [54, 20, 18, 16, 13, 11.9, 10.2, 10.1, 9.77, 9.64, 8.77, 8.44]
    print(len(b))
    x = [i * 10 for i in range(12)]

    plt.plot(x, a, 'r--', label='without warmup')
    plt.plot(x, b, 'g--', label="with warmup")
    plt.xlabel('iterations')
    plt.ylabel("loss")
    plt.legend()
    plt.show()


def parse_refinedet_txt(txt):
    """
    Each line contain iter xx || loss xx || timer
    Args:
        txt:

    Returns:

    """
    result = {}
    with open(txt, 'r') as f:
        result['iter'] = []
        result['arm_l'] = []
        result['arm_c'] = []
        result['odm_l'] = []
        result['dom_c'] = []

        while True:
            line = f.readline()
            if line is None or line is '':
                break
            iter, loss, time = line.split('||')
            loss_list = loss.strip().split(' ')
            result["iter"].append(int(iter.split(' ')[1]))
            result['arm_l'].append(int(loss_list[2]))
            result['arm_c'].append(int(loss_list[5]))
            result['odm_l'].append(int(loss_list[8]))
            result['odm_c'].append(int(loss_list[-1]))

    return result


def loss_plot(info_dict, info_dict2):
    """
    plot loss curve in Refinedet.
    Args:
        info_dict: (dict) contains iter, arm_l, arm_c, odm_l, odm_c

    Returns:

    """
    plt.plot(info_dict['iter'], info_dict['arm_l'], 'r', label='arm_l_scratch')
    plt.plot(info_dict2['iter'], info_dict2['arm_l'], 'r--', label='arm_l_pretrained')


def montage_plot(images, save_path=None, figsize=(30, 10), num_montages=1):
    if num_montages == 1:
        fig = plt.figure(figsize=figsize)
        montage_imgs = montage(images)
        plt.imshow(montage_imgs)
        plt.axis('on')
        plt.title("样本示例")
        if save_path is not None:
            plt.savefig(save_path)

# batch_rgb = montage_rgb(train_x)
# batch_seg = montage(train_y[:, :, :, 0])
# ax1.imshow(batch_rgb)
# ax1.set_title('Images')
# ax2.imshow(batch_seg)
# ax2.set_title('Segmentations')
# ax3.imshow(montage_rgb(mark_image(train_x,train_y)))
# ax3.set_title('Outlined Ships')
# fig.savefig('overview.png')
