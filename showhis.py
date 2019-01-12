import matplotlib.pyplot as plt
fs=20

def showhis(his, title=''):
    plt.cla()
    for key in his.history:
        plt.plot(range(1, 1 + len(his.history[key])), his.history[key], label=key)
    plt.title(title,fontsize=fs)
    plt.xlabel('epoch',fontsize=fs)
    plt.ylabel('loss and loss_val',fontsize=fs)
    plt.legend(fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.tight_layout()
    plt.savefig(title+'.png')
