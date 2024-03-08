from psychopy import visual, event
from psychopy.visual import TextStim
import random


class StroopTrial:
    def __init__(self, win, text, color):
        self.win = win
        self.text = text
        self.color = color
        self.content = []
        self.content.append(TextStim(win, text, color))
        self.congruent = False
        self.congruent = self.text.lower() == self.color.lower()
        self.text_stim = visual.TextStim(window, text=self.text, color=self.color)

    def draw(self):
        self.text_stim.draw()

    def run(self):
        self.draw()
        self.win.flip()
        print(self.congruent)
        event.waitKeys()


window = visual.Window(fullscr=False)
colors = ['red', 'green', 'blue', 'yellow']
for _ in range(12):
    random_text = random.randint(1, 4)-1
    random_color = random.randint(1, 4)-1
    s_trial = StroopTrial(window, text=colors[random_text], color=colors[random_color])
    s_trial.run()
# Close the window
window.close()
