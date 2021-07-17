import matplotlib.pyplot as plt
from IPython import display

#derived from https://github.com/python-engineer/snake-ai-pytorch

plt.ion()

def plot(scores, mean_scores, title):
    display.clear_output(wait=True)
    #display.display(plt.gcf())
    plt.clf()
    plt.title(title)
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    _ = plt.plot(scores)
    _ = plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    _ = plt.show(block=False)
    plt.pause(.1)

def dashboard(initial_config,learning_score, learning, playing_score):
    display.clear_output(wait=True)
    plt.clf()
    plt.title(title)

