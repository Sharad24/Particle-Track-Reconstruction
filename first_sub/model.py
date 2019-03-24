from subprocess import call
import pandas as pd
import os

path = os.path.dirname(os.path.realpath(__file__))


def run(i):
    retval = call(['bash', '-c', './predict %s %s' % (str(i), path)])
    return 1

class Model():
    global path
    def __init__(self):
        super().__init__()
        self.path_main = os.path.join(os.path.dirname(__file__),'main.cpp')
        os.system("g++ " + self.path_main + " -g -std=c++11 -O3 -DEVAL -o predict")

    def predict_one_event(self, event_id, event, cells=None):
        with open(str(event_id) + "-hits.csv", 'w') as file:
            event.to_csv(file, sep=',', index = False)
        if cells is not None:
            with open(str(event_id) + "-cells.csv", 'w') as file:
                cells.to_csv(file, sep=',', index = False)
        run(event_id)
        self.path_submission = os.path.join(path, "submission" + str(event_id) + ".csv")
        sub = pd.read_csv(self.path_submission)
        return sub
