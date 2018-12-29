import matplotlib.pyplot as plt


def showhis(his, title=''):
    for key in his.history:
        plt.plot(range(1, 1 + len(his.history[key])), his.history[key], label=key)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss and loss_val')
    plt.legend()
    plt.savefig(title+'.png')
