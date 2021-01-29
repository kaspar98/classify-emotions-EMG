# Classifying Emotions with EMG
This project was completed as course requirement for Introduction to Neuroscience (administered by University of Tartu).

Electromyography (EMG) is concerned with the reading and analysis of electrical signals produced by the skeletal muscles. This project reads, processes and classifies EMG signals from facial muscles, classifying these signals as: Happy, Sad or Neutral affective states.

A complete writeup of the project is available [here](https://alar-kirikal.medium.com/classifying-emotions-from-emg-ed2c17ddd8bf)

## What this repo contains:
- Preprocessing methods including: 
    - Notch Filtering 
    - Bandpass filtering
    - Standardization
    - Signal windowing, to select samples containing emotional responses recorded
- Feature Extraction methods for signal:
    - amplitude (eg. rms, peak, mavfd)
    - entropy (eg. approximate entropy, sample entropy)
    - variability (eg. range, std)
    - frequency (eg. fmed, fmean)
    - more features for EMG signal [here](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140330)
- Notebooks to generate dataset and classify FE'd data 
- Raw EMG signal data from 8 experiments
- Images used to induce affective states, from OASIS

# Notebooks
Notebooks can be run easily online with binder, as package requirements have been provided. Experiments have been designed with repeatability in mind and main notebooks include:

- Script to generate Image Dataset
- Classifier notebook to show model building steps and final model 

# Recordings

All signal recordings, were done with the [OpenBCI CytonBoard](https://docs.openbci.com/docs/02Cyton/CytonLanding) with electrodes attached to the corrugator supercilii and zygomaticus major as shown below:

![Alt text](https://miro.medium.com/max/566/1*RHuaWZ9oDbTeaFPAl5-EIg.png) 