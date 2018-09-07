var=`ps aux |grep CGRrealtime.py | awk '{print $2}'`
for data in $var
do
	sudo kill -9 $data
done

var=`ps aux |grep RealTimePlotTemplate.py | awk '{print $2}'`
for data in $var
do
	sudo kill -9 $data
done

