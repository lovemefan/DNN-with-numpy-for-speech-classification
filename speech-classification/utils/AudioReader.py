#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @FileName  :AudioReader.py
# @Time      :2022/9/29 16:37
# @Author    :lovemefan
# @email     :lovemefan@outlook.com
import struct
import array
import numpy as np


class AudioReader:
    """
    read audio from sanic request
    """
    def __init__(self):
        pass

    @staticmethod
    def get_info(self, path: str):
        with open(path, 'rb') as f:
            name, data_lengths, _, _, _, _, channels, sample_rate, bit_rate, block_length, sample_bit, _, pcm_length = struct.unpack_from(
                '<4sL4s4sLHHLLHH4sL', f.read(44))
            assert sample_rate == 16000, "sample rate must be 16000"
            nframes = pcm_length // (channels * 2)
        return nframes

    @staticmethod
    def read_pcm16(data):
        """
        convert bytes into array of pcm_s16le data
        :param data: PCM format bytes or str of path
        :return:
        """
        if type(data) is str:
            with open(data, 'rb') as f:
                data = f.read()

        if b"RIFF" != data[:4]:
            raise ValueError(f"WAVE: RIFF header not found")

        shortArray = array.array('h')  # int16
        # header of wav file
        info = data[:44]
        frames = data[44:]
        name, data_lengths, _, _, _, _, channels, sample_rate, bit_rate, block_length, sample_bit, _, pcm_length = struct.unpack_from(
            '<4sL4s4sLHHLLHH4sL', info)
        # shortArray each element is 16bit
        shortArray.frombytes(frames)  # struct.unpack
        data = np.array(shortArray)
        # 缩放
        data = data / np.max(data)
        return data, sample_rate