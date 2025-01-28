for ((i =1; i <= $1; i++));
do
python -u enas_listener.py --proc_id=$i --filename $2 --output_filename $3 &
done
