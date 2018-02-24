#!/bin/bash

# this is a shell script command to automatically download and extract 
# files from Neuran Firing project. 
#
# NOTES : 
#  * this script assumes unzip command is installed



for SMAPLE in 0 1 2 .. 19
do
	printf -v SMAPLE "%02d" $SMAPLE
	wget "https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.00.$SMAPLE.zip"
	unzip -q "neurofinder.00.$SMAPLE.zip" 
	rm "neurofinder.00.$SMAPLE.zip"
done

for SMAPLE in 0 1 2 .. 9
do
	printf -v SMAPLE "%02d" $SMAPLE
	wget "https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.00.$SMAPLE.test.zip"
	unzip -q "neurofinder.00.$SMAPLE.zip" 
	rm "neurofinder.00.$SMAPLE.zip"
done