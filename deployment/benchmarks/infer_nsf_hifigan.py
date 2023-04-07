import time

import numpy as np
import onnxruntime as ort
from scipy.io import wavfile

mel = np.load('deployment/assets/mel.npy')
f0 = np.load('deployment/assets/f0.npy')

print('mel', mel.shape)
print('f0', f0.shape)

session = ort.InferenceSession(
    'deployment/assets/nsf_hifigan.onnx',
    providers=['CPUExecutionProvider']
)

start = time.time()
wav = session.run(['waveform'], {'mel': mel, 'f0': f0})[0]
end = time.time()

print('waveform', wav.shape)
print('cost', end - start)


wavfile.write('deployment/assets/waveform.wav', 44100, wav[0])
