## Content
Custom MATLAB code for manuscript: Y. Su, I. Olasagasti, AL. Giraud, A deep hierarchy of predictions enables assignment of semantic roles in real-time speech comprehension.

The folder `speech stimuli` contains .wav files for single syllables used in the simulation. All syllables are synthesized with Praat (https://www.fon.hum.uva.nl/praat/), using British Male Speaker 1.

The folder 'sentence generation' contains custom MATLAB code to combine single syllables into sentences and convert them to spectral-temporal patterns for the model input. The conversion from sound files to spectral-temporal patterns uses the model of auditory periphery by Chi and Shamma (2005) (https://github.com/tel/NSLtools).

The folder 'speech model' contains custom MATLAB code for model simulation and plotting. This part requires the software package SPM12 (https://www.fil.ion.ucl.ac.uk/spm/software/spm12/).

Yaqing SU (yaqing.su@unige.ch)
