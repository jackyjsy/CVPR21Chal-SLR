MAXX=1
MINX=-2.5
MAXY=1.3125
MINY=-1.3125
ITERATIONS=30

../bin/ImageStack -push 800 600 1 3 -evalchannels "(x/width)*($MAXX - $MINX) + $MINX" "(y/height)*($MAXY - $MINY) + $MINY" 0 -loop $ITERATIONS --evalchannels "[0]*[0] - [1]*[1] + (x/width)*($MAXX - $MINX) + $MINX" "2*[0]*[1] + (y/height)*($MAXY - $MINY) + $MINY" "(([0]*[0] + [1]*[1]) > 2) / $ITERATIONS + [2]" --clamp -50 50 -evalchannels "[2]" 1 "[2]" -normalize -colorconvert hsv rgb -display
