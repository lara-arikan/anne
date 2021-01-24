Barbara is a sonification library for floats, images, graphs and text. She currently operates only on text, and arrays and matrices of floats. 
Being able to sonify arrays of numbers is in itself quite interesting: spectral absorption lines (https://soundcloud.com/vogelin/fraunhofer-lines), daily COVID cases in San Francisco (https://soundcloud.com/vogelin/daily-cases), and yearly grain harvests in the Soviet Union (https://soundcloud.com/vogelin/daily-cases) can all be reduced to lists of floats. 
Barbara's matrix functionality allows us to listen to multiple tracks of data at once, and have those tracks interact.

Using Barbara is very quick. Read in your data as a numpy array and feed it to sonify.floats:

```
from barbara import sonify
import numpy as np

# Amount of time taken per puzzle in chess book:

puzzle_times = np.array([12.5, 12.92, 32.08, 73.4, 25.89, 36.81, 295.22, 16.37, 225.95])
sonify.floats(puzzle_times, 100, 30, filename = 'puzzles.mid')

```

The crucial arguments for this function are the minimum and maximum pitch values. I find that not going below 30 makes for a cleaner listening experience, and that values above 120 hurt the ears. It also seems after some experiment that if one wants to highlight a trend of falls and rises, cutting out certain bass frequencies with below_bound ensures the trend is audible. This is because for a dataset like daily COVID cases, there are days when fewer people get sick during periods when cases are generally rising; letting those bass frequencies sound distracts from this upward motion. Another point is intervals of division; adding rests regularly where a rest would make sense (e.g. the end of a week or month) prevents overwhelm for the listener.

Barbara can also sonify text by tokenizing it, pruning certain words (or not, if the user wishes), and converting it to an array of integers corresponding to the frequency of the token's appearance in the input string. This array is converted to MIDI using the same floats function.


```
buchner = "No, Julie, I love you like the grave."
sonify.freqtext(buchner, 80, 20, filename = 'julie.mid')

```
