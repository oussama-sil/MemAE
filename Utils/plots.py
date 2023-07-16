import matplotlib.pyplot as plt



def plot_loss_evol(losses_train,losses_val,title_figure="Evolution of the loss",save=False,title_save='./figure_evolution_loss'):
    epochs = [i+1 for i in range(len(losses_train))]
    plt.plot(epochs, losses_train, label='Train Loss')
    plt.plot(epochs, losses_val, label='Evaluation Loss')
    plt.legend()
    plt.title(title_figure)
    plt.show()

    if save:
        plt.savefig(title_save)




def plot_metric_evol(metric_val,title_figure="Evolution of the evaluation metric",save=False,title_save='./figure_evolution_metric'):
    epochs = [i+1 for i in range(len(metric_val))]
    plt.plot(epochs, metric_val, label='Evaluation metric evolution')
    plt.legend()
    plt.title(title_figure)
    plt.show()

    if save:
        plt.savefig(title_save)

    pass