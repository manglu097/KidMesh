import matplotlib.pyplot as plt

def plot_loss(log_vals, keys, save_path):
# 假设的log_vals字典，其中包含了10次迭代的损失值
    

    # 提取迭代次数和损失值
    iterations = list(range(1, len(log_vals[f'{keys}']) + 1))  # 生成1到N的迭代次数列表
    loss_values = log_vals[f'{keys}']  # 提取损失值列表

    # 绘制损失曲线
    plt.scatter(iterations, loss_values, label=f'{keys}_Loss', marker='.')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # 保存图像到文件，而不是展示它
    plt.savefig(f'{save_path}/{keys}_loss_curve.png', dpi=300)  # 指定保存路径和文件名，以及分辨率dpi

    # 清理当前绘图环境，避免后续的绘图命令影响当前图像
    plt.clf()
