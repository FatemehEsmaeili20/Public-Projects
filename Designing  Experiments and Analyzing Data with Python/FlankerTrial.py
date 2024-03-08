from psychopy import visual, event
from psychopy.visual import Window
import psychopy.monitors


class FlankerTrial:
    def _init_(self, win, correct='right', trial_type='congruent'):
        self.window = win
        self.correct = correct
        self.trial_type = trial_type
        self.stimuli = []
        print(self.correct)
        # Positions for the flanker stimuli
        positions = [-200, -100, 0, 100, 200]
        trial_image = 'flanker_arrow.png'
        stimulus = None
        # Create ImageStims and add them to the stimuli list
        for position in positions:
            #stimulus = None  # Initialize stimulus variable

            if self.trial_type == 'congruent':
                stimulus = visual.ImageStim(self.window, image=trial_image, flipHoriz=False, pos=(position, 0))
            elif self.trial_type == 'incongruent':
                if position == 0:
                    if self.correct == 'right':
                        stimulus = visual.ImageStim(self.window, image=trial_image, flipHoriz=True, pos=(position, 0))
                    else:
                        stimulus = visual.ImageStim(self.window, image=trial_image, flipHoriz=False, pos=(position, 0))
                else:
                    if self.correct == 'right':
                        stimulus = visual.ImageStim(self.window, image=trial_image, flipHoriz=False, pos=(position, 0))
                    else:
                        stimulus = visual.ImageStim(self.window, image=trial_image, flipHoriz=True, pos=(position, 0))

            elif self.trial_type == 'neutral':
                if position == 0:
                    if self.correct == 'right':
                        stimulus = visual.ImageStim(self.window, image=trial_image, flipHoriz=False, pos=(position, 0))
                    else:
                        stimulus = visual.ImageStim(self.window, image=trial_image, flipHoriz=True, pos=(position, 0))
            print(stimulus)
            self.stimuli.append(stimulus)

    def draw(self):
        for stimulus in self.stimuli:
            stimulus.draw()

    def run(self):
        self.draw()
        self.window.flip()
        event.waitKeys()


window = Window(fullscr=False, units='pix', size=[800, 600])  # Adjust the size as per your requirements
f_trial = FlankerTrial(window, trial_type='incongruent')
f_trial.run()
f_trial = FlankerTrial(window)
f_trial.run()
f_trial = FlankerTrial(window, trial_type='incongruent')
f_trial.run()
f_trial = FlankerTrial(window, trial_type='neutral')
f_trial.run()
f_trial = FlankerTrial(window, correct='left')
f_trial.run()
f_trial = FlankerTrial(window, correct='left', trial_type='incongruent')
f_trial.run()

window.close()