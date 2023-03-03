%%%%%%%%%%% Convert PCM to WAV %%%%%%%%%%%
% This script converts a 16-bit stereo PCM 
% into a WAV file 
% INPUT: 
%   - Number of channels of PCM data
%   - Filename input
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc

num_of_channels = 1;

% Read source PCM
% filename = 'voice_0_test.pcm';            % INPUT FILE NAME
% filename = 'voice_0_202210111726_test.pcm';            % INPUT FILE NAME
filename = 'voice_0_sin_hc.pcm';
fid = fopen(filename);                      % Open raw pcm file
audio = int16(fread(fid, Inf, 'int16'));    % Convert data into 16 bit
fclose(fid);                                % Close pcm file

% Extract Left and Right channel
audio_wav = [];
for ch = 1:num_of_channels
  audio_wav=[audio_wav, audio(ch:num_of_channels:end,1)];
end

% Set Sampling Rate
Fs = 48000;
% Save into WAV format
audiowrite([filename(1:end-4) '_pcm.wav'], audio_wav, Fs,'BitsPerSample', 16); % Write wav

% Play File
y = double(audio_wav)./2^15; % convert int16 data to float(double) -1 to 1
% sound(y,Fs);
