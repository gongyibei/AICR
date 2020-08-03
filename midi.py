import numpy as np
import os
import pretty_midi
from collections import Counter
import mido
from mido import MidiFile
import pickle
import math
import heapq
import pandas


class CSMTMidi(MidiFile):
    def __init__(self, filename):
        # self.filename
        # self.type
        # self.ticks_per_beat
        super().__init__(filename=filename)
        self.pretty_midi = pretty_midi.PrettyMIDI(self.filename)
        self.atonal = self.is_atonal()
        self.notes, self.times, self.start_times, self.total_time = self.analysis(
        )
        self.key = None
        self.correct_notes = None
        self.tempo = self.estimate_tempo()
        self.beats = self.estimate_beats()

    def estimate_tempo(self):
        total_time = self.total_time
        cnt = []
        for time in set(self.times):
            cnt.append(time // round(128 * time / total_time))
        cnt = Counter(cnt)
        # return cnt.most_common(1)[0][0]
        return sum(cnt) // len(cnt)

    def estimate_beats(self):
        beats = []
        for time in self.times:
            beats.append(time // self.tempo)
        return beats

    def is_atonal(self):
        pitch_class_histogram = list(
            self.pretty_midi.get_pitch_class_histogram())
        PITCHS = 0b101010110101
        pitchs = 0
        for i, pitch in enumerate(pitch_class_histogram):
            if pitch != 0:
                n = 1
            else:
                n = 0
            pitchs |= n << i
        for _ in range(12):
            if pitchs | PITCHS == PITCHS:
                return False
            pitchs = (pitchs >> 1) | ((pitchs & 1) << 11)
        return True

    def analysis(self):
        notes = []
        times = []
        start_times = []
        rec = {}
        cur_time = 0
        for msg in self.tracks[1]:
            if 'note' in msg.__dir__():

                if msg.note in rec:
                    cur_time += msg.time
                    notes.append(msg.note)
                    times.append(cur_time - rec[msg.note])
                    rec.pop(msg.note)
                else:
                    start_times.append(cur_time)
                    cur_time += msg.time
                    rec[msg.note] = cur_time
        return notes, times, start_times, cur_time

    def preprocess(self):
        vec = []
        for note, beat in zip(self.notes, self.beats):
            vec += [note] * beat

        if len(vec) < 128:
            vec += [vec[-1]] * (128 - len(vec))
        elif len(vec) > 128:
            vec = vec[:128]
        return np.array(vec)



def load_csmt_midi(filename):
    return CSMTMidi(filename).preprocess().reshape(1, 128)

def load_online_midi1(filename):
    midi = pretty_midi.PrettyMIDI(filename)
    datas = []
    # 采样周期为1/4拍
    T = midi.tick_to_time(midi.resolution) / 4
    # 采样频率
    fs = 1 / T
    for inst in midi.instruments:
        # 去掉鼓轨道
        if inst.is_drum:
            continue
        notes = inst.notes
        # 补齐
        for i in range(len(notes) - 1):
            notes[i].end = notes[i + 1].start

        # piano_roll 类似于midi可视化界面的矩形区域
        piano_roll = inst.get_piano_roll(fs=fs)
        
        # 有多少个 1/4拍
        length = piano_roll.shape[1]
        # 分段，128个最小单位一组，即一个小节
        indices = np.arange(128, length, 128)
        for clip in np.split(piano_roll, indices, axis=1):
            if clip.shape[1] == 128:
                vec = np.argmax(clip, axis=0)
                # 过滤
                if (0 not in set(vec)) and len(set(vec)) > 2:
                    # 1*128 转 128*128
                    data = vec2mat(vec)
                    datas.append(data)
    return np.stack(datas).reshape([len(datas), 1, 128, 128])

def load_online_midi(filename):
    midi = pretty_midi.PrettyMIDI(filename)
    vecs = []
    # 采样周期为1/4拍
    T = midi.tick_to_time(midi.resolution) / 4
    # 采样频率
    fs = 1 / T
    for inst in midi.instruments:
        # 去掉鼓轨道
        if inst.is_drum:
            continue
        notes = inst.notes
        
        # 补齐
        for i in range(len(notes) - 1):
            if notes[i + 1].start - notes[i].end <= 16*T:
                notes[i].end = notes[i + 1].start

        # piano_roll 类似于midi可视化界面的矩形区域
        piano_roll = inst.get_piano_roll(fs=fs).T
        
        vec = []
        i = 0
        for col in piano_roll:
            if np.max(col) != 0:
                break
            i += 1
        
        for col in piano_roll[i:i + 128]:
            max_ = max(col)
            sum_ = sum(col)
   
            if max_ == sum_ and max_ > 0:
                vec.append(max_)
            else:
                break
        if len(vec) < 128:
            break

        if len(set(vec)) > 2:
            # 1*128 转 128*128
            vecs.append(vec)
    if len(vecs) == 0:
        return None
    vec = max(vecs, key = lambda x: len(set(x)))
    return np.array(vec).reshape(1, 128)

if __name__ == '__main__':
    pass