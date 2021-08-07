import matplotlib.pyplot as plt
from IPython import display

#derived from https://github.com/python-engineer/snake-ai-pytorch

plt.ion()

def plot_init(title):
    ##display.clear_output(wait=True)
    #display.display(plt.gcf())
    ##plt.clf()
    #plt.title(title)
    #plt.figure()
    fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(10, 9))
    fig.suptitle(title, fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Games')
    ax1.set_ylabel('Game Score')
    ax2.set_xlabel('Number of Games')
    ax2.set_ylabel('Model Score')
    ax3.set_xlabel('Number of Games')
    ax3.set_ylabel('Learning Rate')
    return (ax1,ax2,ax3)

def plot(axes,scores, mean_scores,model_scores,running_avg_model_scores,learning_rates):
    ax1,ax2,ax3 = axes
    #_ = plt.plot(scores)
    #_ = plt.plot(mean_scores)
    ax1.plot(scores,color='black')
    ax1.plot(mean_scores,color='green')
    #ax1.set_ylim(ymin=0,ymax=60)
    #_ = plt.plot(model_scores)
    ax2.plot(model_scores,color='red')
    ax2.plot(running_avg_model_scores,color='green')
    #ax2.set_ylim(ymin=0,ymax=5000)
    ax3.plot(learning_rates,color='blue')
    #ax3.set_ylim(ymin=0,ymax=0.01)
    #ax1.text(len(scores)-1, scores[-1], str(scores[-1]))
    #ax1.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    #ax2.text(len(model_scores)-1, model_scores[-1], str(model_scores[-1]))
    _ = plt.show(block=False)
    #plt.pause(.1)

def dashboard(initial_config,learning_score, learning, playing_score):
    display.clear_output(wait=True)
    plt.clf()
    plt.title(title)

