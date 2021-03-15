ImageStack -push 512 512 1 1 -noise 0 1 -threshold 0.75 -loop 1000 --convolve 3 3 1 1 1 1 1 0.5 1 1 1 1 zero --eval "(val < 3.75) * (val > 2.25)" --display
