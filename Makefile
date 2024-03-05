FILES= model/func_impl.py \
	   data/data_parallel_preprocess.py

handin.tar: $(FILES)
	tar cvf handin.tar $(FILES)

clean:
	rm -f *~ handin.tar