#!/usr/bin/env python
# coding: utf-8

# In[54]:


from midiutil.MidiFile import MIDIFile


# In[55]:


# Error functions

def validate_values(val, maxf, minf, maxval, minval):
    if maxf < minf or maxf > 127 or minf > 127 or maxf < 0 or minf < 0 or val > maxval or val < minval:
        return False
    
    return True

def validate_types(val, maxf, minf, data):
    if type(val) is not float and type(val) is not int:
        return False
    elif type(maxf) is not int or type(minf) is not int:
        return False
    elif type(data) is not list:
        return False
    return True


# In[56]:


# Additions to data

def add_dividers(pitches: list, interval: int) -> list:
    """
    Adds dividing sounds to a list of pitches at regular intervals.
    This is useful for when the data is periodic (e.g. temperature by month
    for multiple years), or when it is so long that rests may help with
    easier listening.

    :param pitches: prepared list of pitches
    :param interval: integer index (if 5, a divider will be added after every fifth note, etc.)
    :return: altered list of pitches
    """
    
    new_pitches = []
    
    for i, f in enumerate(pitches):
        new_pitches.append(f)
        if i % interval == interval - 1:
            new_pitches.append('div')

    return new_pitches


# In[63]:


# Sonification: minmax feature mapping

def pitch(val: float or int, maxf: int, minf: int, maxval: float, minval: float) -> int:
    """
    Converts a single value in a sequence of data to the MIDI
    pitch that would represent it if the whole sequence were
    sonified using min-max feature mapping between the data points 
    and pitches in the specified interval.
    
    Having this function independently of the full sequence conversion
    helps the user determine bounds and exact pitch values for special
    cases, as when data points above, below or equal to a value will be
    represented by sounds of different volume or duration.
    
    :param val: float value of interest
    :param maxf: maximum pitch, at most 127
    :param minf: minimum pitch, at least 0
    :param maxval: maximum value in data list
    :param minval: minimum value in data list
    :return: MIDI pitch in given interval
    """
    
    # check for errors
    
    if not validate_types_minmax(val, maxf, minf, data):
        raise TypeError("Ensure parameter types match documentation.")
        
    if not validate_values_minmax(val, maxf, minf, maxval, minval):
        raise ValueError("Define minimum and maximum MIDI pitches such that 0 =< minimum < maximum =< 127.")
    
    # use formula for min-max feature mapping
    
    return int((maxf - minf) * (val - minval) / (maxval - minval) + minf)

def pitches(maxf: int, minf: int, maxval: float, minval: float, data: list) -> list:
    """
    Given list of data points in order from first to last,
    such that the values could be mapped onto the y-axis
    with their index in the list as the corresponding x-value,
    this function returns a list of MIDI pitches between the given
    maximum and minimum pitches. This method uses min-max 
    feature mapping between the data points 
    and pitches in the specified interval.
    
    :param maxf: maximum pitch, at most 127
    :param minf: minimum pitch, at least 0
    :param maxval: maximum value in data list
    :param minval: minimum value in data list
    :param data: list of values
    :return: list of pitches ranging between minf and maxf
    """
    
    return [pitch(val, maxf, minf, maxval, minval) for val in data]

def sonify_list(data: list, maxf: int, minf: int, 
                  tempo = 120, volume = 90, duration = 0.25, filename = 'sonified.mid', 
                  below_bound = None, below_volume = 90, below_duration = 0.25, 
                  above_bound = None, above_volume = 90, above_duration = 0.25,
                  equal_val = None, equal_volume = 90, equal_duration = 0.25, 
                  interval = None, int_pitch = 50, int_volume = 0, int_duration = 0.5):
    
    """
    Given list of data points in order from first to last,
    such that the values could be mapped onto the y-axis
    with their index in the list as the corresponding x-value,
    this function creates a MIDI file of notes with pitches
    of magnitude corresponding to the magnitude of the data
    point they represent. These pitches lie between the given
    maximum and minimum pitches. This method uses min-max 
    feature mapping between the data points 
    and pitches in the specified interval.
    
    If the user wishes, they can create exceptions for values
    above or below certain bounds, or equal to some number;
    these exceptional notes may have different durations or
    volumes, depending on how the user wants to emphasize or
    diminish their presence in the final product.
    
    They may also insert rests or set sounds at regular intervals
    to emphasize periodic patterns or to provide an easier listening experience.
    
    :param filename: string + '.mid', the name of the MIDI file to be saved. If unspecified, multiple uses
    of this function will overwrite previous sonification attempts, since they all use a default filename.
    :param data: list of values
    :param maxf: maximum pitch, at most 127
    :param minf: minimum pitch, at least 0
    :param tempo: tempo of MIDI playback, by default 120
    :param volume: volume of notes, by default 90
    :param duration: duration of notes, by default 0.25 (quarter note; convenient for long lists)
    :param below_bound: values below this may be handled as special cases, with different volume or duration.
    Set by default to None. Must specify float.
    :param above_bound: values above this may be handled as special cases, with different volume or duration.
    Set by default to None. Must specify float.
    :param equal_val: values matching this may be handled as special cases, with different volume or duration.
    Set by default to None. Must specify float.
    :params below_volume, above_volume, equal_volume: for special cases, exceptional volumes.
    By default same as other values.
    :params below_duration, above_duration, equal_duration: for special cases, exceptional durations.
    By default same as other values.
    :param interval: index at which dividing sound will be inserted.
    Set by default to None. Must specify int.
    :params int_pitch, int_volume, int_duration: the dividing sound is by default a rest twice as long
    as the other notes, but this can be altered.
    """
    
    # Create MIDI stream of one track, starting from beat 0
    
    midi = MIDIFile(1)
    track = 0
    channel = 0
    time = 0
    midi.addTempo(track, time, tempo)
    
    # Get pitches by feature mapping
    
    maxval = max(data)
    minval = min(data)
    fs = pitches(maxf, minf, maxval, minval, data)
    
    if interval is not None:
        fs = add_dividers(fs, interval)
    
    # Populate MIDI stream

    for f in fs:
        if f == 'div':
            midi.addNote(track, channel, int_pitch, time, int_duration, int_volume)
            time += int_duration
        elif below_bound is not None and f < pitch(below_bound, maxf, minf, maxval, minval):
            midi.addNote(track, channel, f, time, below_duration, below_volume)
            time += below_duration
        elif above_bound is not None and f > pitch(above_bound, maxf, minf, maxval, minval):
            midi.addNote(track, channel, f, time, above_duration, above_volume)
            time += above_duration
        elif equal_val is not None and f == pitch(equal_val, maxf, minf, maxval, minval):
            midi.addNote(track, channel, f, time, equal_duration, equal_volume)
            time += above_duration
        else:
            midi.addNote(track, channel, f, time, duration, volume)
            time += duration
            
    # Save MIDI file to working directory

    with open(filename, "wb") as output_file:
        midi.writeFile(output_file)

