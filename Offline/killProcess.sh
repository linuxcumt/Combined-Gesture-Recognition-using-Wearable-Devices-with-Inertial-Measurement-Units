var=`ps aux |grep QTofflineRealLine.py | awk '{print $2}'`
for data in $var
do
	sudo kill -9 $data
done
clear
