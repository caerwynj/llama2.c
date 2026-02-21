mount -c tcp!127.0.0.1!6701 /n/llm
cat /n/llm/clone > /tmp/fid
fid=`{cat /tmp/fid}
echo "clone gives fid " $fid
echo "temp 0.8" > /n/llm/$fid/ctl
echo -n "Once upon a time" > /n/llm/$fid/data
cat /n/llm/$fid/data
echo "Done"
