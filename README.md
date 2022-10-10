# PS2F
Code for "Ghanekar, Bhargav, et al. "PS $^{2} $ F: Polarized Spiral Point Spread Function for Single-Shot 3D Sensing." IEEE Transactions on Pattern Analysis and Machine Intelligence (2022)."

Demo run: 
python recon_3dvol.py -d test_circ_v1.mat -p rotpsf_2c.mat -rl1 0.01 -rtv 0.00 -lr 0.001 -n 1000 -noise 0.01 --display

Additional functionality: 
Added --dip flag to add Deep Image Prior for reconstruction. Based on paper and code from https://github.com/Waller-Lab/UDN
