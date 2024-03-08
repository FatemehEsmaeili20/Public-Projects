import psychopy
from psychopy import visual, event
from psychopy.visual import Window, TextStim
from psychopy.core import Clock
import pandas as pd
import random
import psychopy.gui as psygui
import os


class Block:
    def __init__(self, win, trial_specification, first_parameter, second_parameter):
        self.win = win
        self.trials = trial_specification
        self.stroop_text = '-'
        self.stroop_color = '-'
        self.flanker_direction = '-'
        self.flanker_congruency = '-'
        self.correct = first_parameter
        self.response = []
        self.user_response = ''
        self.is_correct = False
        self.response_time = 0
        if self.trials == 'StroopTrial':
            self.stroop_text = first_parameter
            self.stroop_color = second_parameter
            self.content = []
            self.content.append(TextStim(win, self.stroop_text, self.stroop_color))
            self.congruent = False
            self.congruent = self.stroop_text.lower() == self.stroop_color.lower()
            self.text_stim = visual.TextStim(win, text=self.stroop_text, color=self.stroop_color)
        elif self.trials == 'FlankerTrial':
            self.flanker_direction = first_parameter
            self.flanker_congruency = second_parameter
            self.stimuli = []
            # Positions for the flanker stimuli
            positions = [-200, -100, 0, 100, 200]
            # Create ImageStims and add them to the stimuli list
            for position in positions:
                # Set the stimulus image based on trial typeb
                if self.flanker_congruency == 'congruent':
                    stimulus = visual.ImageStim(self.win, image='flanker_arrow.png', flipHoriz=False, pos=(position, 0))
                elif self.flanker_congruency == 'incongruent':
                    if position == 0:
                        if self.flanker_direction == 'right':
                            stimulus = visual.ImageStim(self.win, image='flanker_arrow.png', flipHoriz=True, pos=(position, 0))
                        else:
                            stimulus = visual.ImageStim(self.win, image='flanker_arrow.png', flipHoriz=False, pos=(position, 0))
                        self.stimuli.append(stimulus)
                    else:
                        if self.flanker_direction == 'right':
                            stimulus = visual.ImageStim(self.win, image='flanker_arrow.png', flipHoriz=False, pos=(position, 0))
                        else:
                            stimulus = visual.ImageStim(self.win, image='flanker_arrow.png', flipHoriz=True, pos=(position, 0))
                        self.stimuli.append(stimulus)
                elif self.flanker_congruency == 'neutral':
                    if position == 0:
                        if self.flanker_direction == 'right':
                            stimulus = visual.ImageStim(self.win, image='flanker_arrow.png', flipHoriz=False, pos=(position, 0))
                        else:
                            stimulus = visual.ImageStim(self.win, image='flanker_arrow.png', flipHoriz=True, pos=(position, 0))
                        self.stimuli.append(stimulus)

    def wait(self, time):
        #self.win.cleanup()
        psychopy.core.wait(time)

    def draw(self):
        if self.trials == 'StroopTrial':
            self.text_stim.draw()
        if self.trials == 'FlankerTrial':
            for stimulus in self.stimuli:
                stimulus.draw()

    def validate_response(self):
        if self.trials == 'StroopTrial':
            if str(self.response[0]) == 'b':
                self.user_response = 'blue'
            if str(self.response[0]) == 'r':
                self.user_response = 'red'
            if str(self.response[0]) == 'y':
                self.user_response = 'yellow'
            if str(self.response[0]) == 'g':
                self.user_response = 'green'
            if self.user_response == str(self.stroop_text):
                self.is_correct = True
            else:
                self.is_correct = False
        if self.trials == 'FlankerTrial':
            if str(self.response[0]) == 'f':
                self.user_response = 'left'
            if str(self.response[0]) == 'j':
                self.user_response = 'right'
            if self.user_response == str(self.flanker_direction):
                self.is_correct = True
            else:
                self.is_correct = False

    def run(self):
        self.wait(0.1)
        self.draw()
        self.win.flip()
        clock.reset()
        if self.trials == 'StroopTrial':
            self.response = psychopy.event.waitKeys(keyList=['b', 'r', 'y', 'g'])
        if self.trials == 'FlankerTrial':
            self.response = psychopy.event.waitKeys(keyList=['f', 'j'])
        self.response_time = clock.getTime()
        self.validate_response()


def instruction(win, trial_sp):
    if trial_sp == 'StroopTrial':
        text = """
    Now! Stroop Trial will be started! " 
                
    "Press 'b' Key if you mean the color of text is 'Blue'" 
               
    "Press 'r' Key if you mean the color of text is 'Red'" 
               
    "Press 'y' Key if you mean the color of text is 'Yellow'" 
               
    "Press 'g' Key if you mean the color of text is 'Green'" 
               
    "Press any key to Start"
    """

    if trial_sp == 'FlankerTrial':
        text = """
    "Now! Flanker Trial will be started! " 
               
    "Press 'f' Key if you want to show 'left' direction"
               
    "Press 'j' Key if you want to show 'Right' direction" 
               
    "press any key to Start"
    """
    text_stim = visual.TextStim(win, text=text, color='white')
    text_stim.draw()
    win.flip()
    event.waitKeys()


def dump_data(data_list):
    file_name = "output.csv"
    if os.path.exists(file_name):
        include_header = True
    else:
        include_header = False
    df = pd.DataFrame(data_list)
    # Write DataFrame to a CSV file
    df.to_csv(file_name, mode='a', header=include_header, index=False)


def append_stroop_blocks(win, blocks, st_trial):
    trial_sp = 'StroopTrial'
    instruction(win, trial_sp)
    data_list = []
    shuffled_indices = list(st_trial.index)
    random.shuffle(shuffled_indices)
    for index in shuffled_indices:
        first_pa = st_trial.loc[index, 'text']
        second_pa = st_trial.loc[index, 'color']
        blocks.append([first_pa, second_pa])
        B_trial = Block(win, trial_sp, first_pa, second_pa)
        B_trial.run()
        user_d = [str(block_csv), str(user_id), str(user_type), str(B_trial.trials), str(B_trial.flanker_direction),
                  str(B_trial.flanker_congruency), str(B_trial.stroop_text), str(B_trial.stroop_color), str(B_trial.correct),
                  str(B_trial.response),str(B_trial.is_correct), str(B_trial.response_time)]
        data_list.append(user_d)
    dump_data(data_list)


def append_flanker_blocks(win, blocks, fl_trial):
    trial_sp = 'FlankerTrial'
    instruction(win, trial_sp)
    data_list = []
    shuffled_indices = list(fl_trial.index)
    random.shuffle(shuffled_indices)
    for index in shuffled_indices:
        first_pa = fl_trial.loc[index, 'correct']
        second_pa = fl_trial.loc[index, 'type']
        blocks.append([first_pa, second_pa])
        B_trial = Block(win, trial_sp, first_pa, second_pa)
        B_trial.run()
        user_d = [str(block_csv), str(user_id), str(user_type), str(B_trial.trials), str(B_trial.flanker_direction),
                  str(B_trial.flanker_congruency), str(B_trial.stroop_text), str(B_trial.stroop_color), str(B_trial.correct), str(B_trial.response), str(B_trial.is_correct), str(B_trial.response_time)]
        data_list.append(user_d)
    dump_data(data_list)

global user_id
global user_type
global block_csv
global clock
gui = psygui.Dlg(title="Stroop / Flanker study")
gui.addField("Select a test", choices=["Stroop Test", "Flanker Test"])
gui.show()
if gui.OK:
    test_selection = gui.data[0]
    print("You selected:", test_selection)
else:
    print("Test selection canceled by the user")

gui = psygui.Dlg(title="Participant Information")
gui.addText("Participant ID:")
gui.addField("", tip="Enter participant ID")
gui.addText("Participant Type:")
gui.addField("Select a type", choices=["main", "pilot"])
gui.show()
if gui.OK:
    user_id = gui.data[0]
    user_type = gui.data[1]
    print("Participant ID:", user_id)
    print("Participant Type:", user_type)
else:
    print("Participant information canceled by the user")

window = Window(fullscr=False, units='pix', size=[640, 480])
clock = Clock()
st_files = []
fl_files = []
for i1 in range(1, 4):
    st_file_name = f"st_b{i1}.csv"
    st_files.append(st_file_name)
    ft_file_name = f"ft_b{i1}.csv"
    fl_files.append(ft_file_name)
output_file_name = "output.csv"
columns = ['block_csv','User_Id','Test_Type','Test_Name','Flanker_Direction','Flanker_Congruency','Stroop_Text','Stroop_Color','Correct',
            'Response','Is_Correct?','Time']
with open(output_file_name, 'w') as file:
    file.write(','.join(columns))
blocks = []
if test_selection == "Stroop Test":
    for i in range(0, 3):
        st_trial = pd.read_csv(st_files[i])
        block_csv = st_files[i]
        append_stroop_blocks(window, blocks, st_trial)

if test_selection == "Flanker Test":
    for i in range(0, 3):
        fl_trial = pd.read_csv(fl_files[i])
        block_csv = fl_files[i]
        append_flanker_blocks(window, blocks, fl_trial)
# Close the window
window.close()
