# Stability of weights due to thalamus based switching of cortical neurons as in Rikhye, Gilra and Halassa, Nature Neuroscience 2018, Fig. 5b-d and Supplementary Fig. 16b-d.
  
## Installation:  
You need to install python, numpy, scipy and matplotlib. The simulations were run on Ubuntu 18.04 on a server with CPU: Intel Xeon E5-2680 v3 (Haswell) 2x 12 cores, 24 threads, 2.5 GHz, 30 MB cache.  
The code only uses 1 core and takes around 30 min for the task in Fig 5b-d, and around 1 hour for the xor task in Suppl. Fig 16b-d (for one network instance).   
  
## Running an instance:  
First `cd` to the directory in which you've clone this github repo.  
Then `mkdir dataPFCMD` where the data files will be saved.  
Run `python reservoir_PFCMD_simple.py`. By default, it runs with MD present on the standard task in the paper (Fig 5b-d).  
This runs 1000 cycles of context 1, 1000 cycles of context 2, and then 200 cycles of context 1 again (a cycle is one presentation of all cues of the current context).  
This will generate 4 figures: (a) cues, a few reservoir PFC neural activity for each cue subset, MD neural activity and output activity vs time; (b) mean squared error versus cycle num; (c) evolution of a few output weights for cues A and B versus cycle number; and (d) mean activity of each neuron during a few context 1 cycles.   
  
To turn MD off, set `self.MDstrength = 0.` instead of `1.` in reservoir_PFCMD_simple.py.  
To run the xor task (Suppl Fig 16b-d), set `self.xorTask = True` in reservoir_PFCMD_simple.py.  
  
## To reproduce Figure 5b-d:  
1. Set seeds 0 to 9 in reservoir_PFCMD_simple.py by modifying line: `self.RNGSEED = 0`.  
2. Run twice for each seed setting `self.MDstrength = 0.` and `self.MDstrength = 1.`.  
3. Finally, run `python plot_PFCMD_figs.py` which is save a fig_paper.eps file corresponding to Fig 5b-d.  
  
## To reproduce Suppl. Figure 16b-d:  
Repeat as above, except set `self.xorTask = True` in reservoir_PFCMD_simple.py and set `xorStr = '_xor'` in plot_PFCMD_figs.py.  
  
