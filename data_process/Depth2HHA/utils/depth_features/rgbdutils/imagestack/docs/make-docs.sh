#!/usr/bin/bash
cat header.inc > index.html
#echo "<html><head><title>ImageStack Help</title></head>" > index.html
#echo "<body>" >> index.html
echo "<center><H2>ImageStack Operation Listing</H2></center>" >> index.html
ImageStack -help | tail -n2 | sed "s/-\([^ ]*\)/<a href=\"#\1\">-\1<\/a>/g" >> index.html
OPS=`ImageStack -help | grep "^-" | tail -n1`
for OPERATION in $OPS; do
    OPERATION=${OPERATION/-/}
    echo "<p><a name=\"${OPERATION}\"><h3>${OPERATION}</h3></a>" >> index.html; 
    echo "<p>" >> index.html; 
    ImageStack -help ${OPERATION} | grep -v "Performing operation -help" | sed "s/^$/<p>/" | sed "s/\\$//g" >> index.html; 
    echo "<p>" >> index.html; 
done
echo "</td></tr></body></html>" >> index.html



