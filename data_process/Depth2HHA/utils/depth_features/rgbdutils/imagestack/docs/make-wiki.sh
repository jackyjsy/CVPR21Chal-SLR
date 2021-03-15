#!/usr/bin/bash

# Make the operation listing page
echo "#labels Featured" > OperationIndex.wiki
echo "==!ImageStack Operation Listing==" >> OperationIndex.wiki
echo "_This page was auto-generated from the !ImageStack -help operator. Direct edits to it will be lost._" >> OperationIndex.wiki 

# Make each wiki page
OPS=`ImageStack -help | grep "^-" | tail -n1`
for OPERATION in $OPS; do
    OPERATION=${OPERATION/-/}
    echo $OPERATION
    echo "  * [${OPERATION}]" >> OperationIndex.wiki
    echo "----" > ${OPERATION}.wiki; 
    echo "_This page was auto-generated from the !ImageStack -help operator. Direct edits to it will be lost. Use the comments below to discuss this operation._" >> ${OPERATION}.wiki 
    echo "----" >> ${OPERATION}.wiki 
    echo "{{{" >> ${OPERATION}.wiki
    echo "ImageStack -help ${OPERATION}" >> ${OPERATION}.wiki
    ImageStack -help ${OPERATION} | grep -v "Performing operation -help" >> ${OPERATION}.wiki 
    echo "}}}" >> ${OPERATION}.wiki; 
done
