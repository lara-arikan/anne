# -----------------------------------------------------------------------------
# Name:        floats.py
# Purpose:     Sonification of lists of floats
#
# Author:      Lara Arikan (arikan -at- stanford.edu)
#
# Created:     1/7/2021
# Updated:     1/23/2021
# License:     MIT
# -----------------------------------------------------------------------------

from midiutil.MidiFile import MIDIFile
import numpy as np
import re

# Error functions

def validate_values(val, maxf, minf, maxval, minval):
    if maxf < minf or maxf > 127 or minf > 127 or maxf < 0 or minf < 0 or val > maxval or val < minval:
        return False
    
    return True


# Alterations to data

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

def compress(data: np.array, to_mean: int):
    """
    For a matrix with more rows than is convenient to sonify
    with superimposed tracks (usually more than five rows),
    the user has the option to compress the rows of the matrix
    to fewer subarrays by taking the mean of each successive
    subgroup of to_mean arrays. 
    
    This function is completely sourced from swenzel on StackOverflow:
    https://stackoverflow.com/questions/30379311/fast-way-to-take-average-of-every-n-rows-in-a-npy-array/30379509
    """
    
    compressed = np.cumsum(data, 0)
    result = compressed[to_mean - 1 :: to_mean] / float(to_mean)
    result[1:] = result[1:] - result[:-1]

    remainder = data.shape[0] % to_mean
    
    if remainder != 0:
        if remainder < data.shape[0]:
            lastAvg = (compressed[-1] - compressed [-1 - remainder]) / float(remainder)
        else:
            lastAvg = compressed[-1] / float(remainder)
            
        result = np.vstack([result, lastAvg])
        
    return result

# -----------------------------------------------------------------------------

# Float sonification

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
        
    if not validate_values(val, maxf, minf, maxval, minval):
        raise ValueError("Define minimum and maximum MIDI pitches such that 0 =< minimum < maximum =< 127.")
    
    # use formula for min-max feature mapping
    
    return int((maxf - minf) * (val - minval) / (maxval - minval) + minf)

def pitches(maxf: int, minf: int, maxval: float, minval: float, data: np.array) -> list:
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


# MIDI tools

def populate_track(midi: MIDIFile, track: int, channel: int, fs: list,
                  tempo: int, volume: int, duration: int, time: float,
                  below_bound, below_volume: int, below_duration: float, 
                  above_bound, above_volume: int, above_duration: float,
                  equal_val, equal_volume: int, equal_duration: float, 
                  interval, int_pitch: int, int_volume: int, int_duration: float):
    
    # Different tracks may have different tempos
    
    midi.addTempo(track, time, tempo)
    
    # Populate with pitches
    
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
            
    return midi


# 1D array of floats! Special case of matrix

def floats(data: np.array, maxf: int, minf: int, 
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
    
    # Get pitches by feature mapping
    
    maxval = np.max(data)
    minval = np.min(data)
    fs = pitches(maxf, minf, maxval, minval, data)
    
    if interval is not None:
        fs = add_dividers(fs, interval)
    
    # Populate MIDI stream

    midi = populate_track(midi, track, channel, fs,
                  tempo, volume, duration, time,
                  below_bound, below_volume, below_duration, 
                  above_bound, above_volume, above_duration,
                  equal_val, equal_volume, equal_duration, 
                  interval, int_pitch, int_volume, int_duration)
            
    # Save MIDI file to working directory

    with open(filename, "wb") as output_file:
        midi.writeFile(output_file)


# Multi-row data into multi-track MIDI

def matrix(data: np.array, maxf: int, minf: int, 
           tempo = 120, volume = 90, duration = 0.25, filename = 'sonified.mid', 
           below_bound = None, below_volume = 90, below_duration = 0.25, 
           above_bound = None, above_volume = 90, above_duration = 0.25,
           equal_val = None, equal_volume = 90, equal_duration = 0.25, 
           interval = None, int_pitch = 50, int_volume = 0, int_duration = 0.5, 
           only_max = False, only_min = False, to_mean = 0):
    
    """
    Given array of arrays of of data points in order from first to last,
    such that the values could be mapped onto the y-axis
    with their index in the array as the corresponding x-value,
    this function creates a multi-track MIDI file where each track
    is comprised of notes with pitches of magnitude 
    corresponding to the magnitude of the data point they represent. 
    These pitches lie between the given maximum and minimum pitches. 
    This method uses min-max feature mapping between the data points 
    and pitches in the specified interval.
    
    The difference of this function from floats is that it allows
    the user to input multiple data streams at the same time, and outputs
    a multi-track MIDI file where each track is a data stream processed 
    exactly as a 1D array would be. To declutter the sound, the user may
    use the means functionality to average each grouping of n consecutive rows,
    such that there are fewer MIDI tracks being listened to at once.
    
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
    :param to_mean: how many consecutive rows to average
    :params only_min, only_max: of stacked rows, get only minimum or maximum values and process that 1D array 
    of mins or maxes instead.
    """
    
    # Determine how many tracks
    
    if to_mean != 0:
        data = compress(data, to_mean)
    
    if only_min and only_max:
        raise ValueError("The program cannot take only minimum and only maximum values at once.")
        
    if only_min:
        data = data.min(0)
        
    if only_max:
        data = data.max(0)
        
    ntracks = data.shape[0]
    
    if ntracks == 0:
        raise ValueError("You can't sonify an empty array.")
    
    if ntracks == 1:
        return floats(data, maxf, minf, 
           tempo, volume, duration, filename, 
           below_bound, below_volume, below_duration, 
           above_bound, above_volume, above_duration,
           equal_val, equal_volume, equal_duration, 
           interval, int_pitch, int_volume, int_duration)
         
    # Create MIDI stream of n tracks, starting from beat 0
    
    midi = MIDIFile(ntracks)
    channel = 0
    time = 0
    
    # Populate MIDI stream

    for track, stream in enumerate(data):
        
        # Get pitches by feature mapping

        maxval = np.max(stream)
        minval = np.min(stream)
        fs = pitches(maxf, minf, maxval, minval, stream)

        if interval is not None:
            fs = add_dividers(fs, interval)
        
        midi = populate_track(midi, track, channel, fs,
                  tempo, volume, duration, time,
                  below_bound, below_volume, below_duration, 
                  above_bound, above_volume, above_duration,
                  equal_val, equal_volume, equal_duration, 
                  interval, int_pitch, int_volume, int_duration)
        
    # Save MIDI file to working directory

    with open(filename, "wb") as output_file:
        midi.writeFile(output_file)

# -----------------------------------------------------------------------------

# Text sonification

# All stopwords: Pruning these allows greater discrimination between texts. 
# Culled from https://programminghistorian.org/en/lessons/counting-frequencies 

stopwords = ['a', 'about', 'above', 'across', 'after', 'afterwards']
stopwords += ['again', 'against', 'all', 'almost', 'alone', 'along']
stopwords += ['already', 'also', 'although', 'always', 'am', 'among']
stopwords += ['amongst', 'amoungst', 'amount', 'an', 'and', 'another']
stopwords += ['any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere']
stopwords += ['are', 'around', 'as', 'at', 'back', 'be', 'became']
stopwords += ['because', 'become', 'becomes', 'becoming', 'been']
stopwords += ['before', 'beforehand', 'behind', 'being', 'below']
stopwords += ['beside', 'besides', 'between', 'beyond', 'bill', 'both']
stopwords += ['bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant']
stopwords += ['co', 'computer', 'con', 'could', 'couldnt', 'cry', 'de']
stopwords += ['describe', 'detail', 'did', 'do', 'done', 'down', 'due']
stopwords += ['during', 'each', 'eg', 'eight', 'either', 'eleven', 'else']
stopwords += ['elsewhere', 'empty', 'enough', 'etc', 'even', 'ever']
stopwords += ['every', 'everyone', 'everything', 'everywhere', 'except']
stopwords += ['few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first']
stopwords += ['five', 'for', 'former', 'formerly', 'forty', 'found']
stopwords += ['four', 'from', 'front', 'full', 'further', 'get', 'give']
stopwords += ['go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her']
stopwords += ['here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers']
stopwords += ['herself', 'him', 'himself', 'his', 'how', 'however']
stopwords += ['hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed']
stopwords += ['interest', 'into', 'is', 'it', 'its', 'itself', 'keep']
stopwords += ['last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made']
stopwords += ['many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine']
stopwords += ['more', 'moreover', 'most', 'mostly', 'move', 'much']
stopwords += ['must', 'my', 'myself', 'name', 'namely', 'neither', 'never']
stopwords += ['nevertheless', 'next', 'nine', 'no', 'nobody', 'none']
stopwords += ['noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of']
stopwords += ['off', 'often', 'on','once', 'one', 'only', 'onto', 'or']
stopwords += ['other', 'others', 'otherwise', 'our', 'ours', 'ourselves']
stopwords += ['out', 'over', 'own', 'part', 'per', 'perhaps', 'please']
stopwords += ['put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed']
stopwords += ['seeming', 'seems', 'serious', 'several', 'she', 'should']
stopwords += ['show', 'side', 'since', 'sincere', 'six', 'sixty', 'so']
stopwords += ['some', 'somehow', 'someone', 'something', 'sometime']
stopwords += ['sometimes', 'somewhere', 'still', 'such', 'system', 'take']
stopwords += ['ten', 'than', 'that', 'the', 'their', 'them', 'themselves']
stopwords += ['then', 'thence', 'there', 'thereafter', 'thereby']
stopwords += ['therefore', 'therein', 'thereupon', 'these', 'they']
stopwords += ['thick', 'thin', 'third', 'this', 'those', 'though', 'three']
stopwords += ['three', 'through', 'throughout', 'thru', 'thus', 'to']
stopwords += ['together', 'too', 'top', 'toward', 'towards', 'twelve']
stopwords += ['twenty', 'two', 'un', 'under', 'until', 'up', 'upon']
stopwords += ['us', 'very', 'via', 'was', 'we', 'well', 'were', 'what']
stopwords += ['whatever', 'when', 'whence', 'whenever', 'where']
stopwords += ['whereafter', 'whereas', 'whereby', 'wherein', 'whereupon']
stopwords += ['wherever', 'whether', 'which', 'while', 'whither', 'who']
stopwords += ['whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with']
stopwords += ['within', 'without', 'would', 'yet', 'you', 'your']
stopwords += ['yours', 'yourself', 'yourselves']

# Standardize string for counting

def process(s: str):  
    s = re.sub(r'[^\w\s]', '', s) # remove punctuation
    return s.lower() # make string lowercase

# Build dictionary of word to frequency

def freqdict(tokens: list):
    wordfreq = {}
    
    for token in tokens:
        wordfreq[token] = tokens.count(token)
        
    return wordfreq

def freqtext(text: str, maxf: int, minf: int, 
             stops = stopwords, 
             tempo = 120, volume = 90, duration = 0.25, filename = 'sonified.mid', 
             below_bound = None, below_volume = 90, below_duration = 0.25, 
             above_bound = None, above_volume = 90, above_duration = 0.25,
             equal_val = None, equal_volume = 90, equal_duration = 0.25, 
             interval = None, int_pitch = 50, int_volume = 0, int_duration = 0.5):
    
    """
    This function takes in text as a string, tokenizes it, then replaces each token
    with the frequency of its appearance in the text, which produces a 1D array of ints.
    This can be sonified using floats, documented above.
    
    For long texts like stories or articles, it will pay to strip away what are known
    as stopwords: words that appear often in language regardless of context, like pronouns
    or prepositions. This helps discriminate between texts of different subject matter.
    The user can supply their own list of stopwords, or set this variable to None if they
    have a shorter string they'd like not to prune.
    
    :param stops: a list of words that the user does not want to include in the frequency
    count; by default this is a standard list of stopwords.

    """
    tokens = [process(s) for s in text.split()]
    
    if stops is not None:
        tokens = [word for word in tokens if word not in stops]
    
    wordfreq = freqdict(tokens)
    
    freqs = np.array([wordfreq[word] for word in tokens])
        
    return floats(freqs, maxf, minf, 
           tempo, volume, duration, filename, 
           below_bound, below_volume, below_duration, 
           above_bound, above_volume, above_duration,
           equal_val, equal_volume, equal_duration, 
           interval, int_pitch, int_volume, int_duration)