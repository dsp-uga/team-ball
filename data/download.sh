#!/bin/bash

# this is a shell script command to automatically download and extract 
# files from Neuran Firing project. 
#
# NOTES : 
#  * this script assumes unzip command is installed



for SMAPLE in 0 1 2 .. 19
do
	printf -v NUM "%02d" $SMAPLE
	wget "https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.00.$NUM.zip"
	unzip -q "neurofinder.00.$NUM.zip" 
	rm "neurofinder.00.$NUM.zip"
done


wget "https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.all.test.zip"
unzip -q "neurofinder.all.test.zip"
rm "neurofinder.all.test.zip"


# for SMAPLE in 0 1 2 .. 9
# do
# 	printf -v NUM "%02d" $SMAPLE
# 	wget "https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.00.$NUM.test.zip"
# 	unzip -q "neurofinder.00.$NUM.test.zip" 

# 	rm "neurofinder.00.$NUM.test.zip"
# done