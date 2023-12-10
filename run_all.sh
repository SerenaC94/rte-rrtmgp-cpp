cd build && cmake .. -DSYST=ubuntu -DUSECUDA=on && make -j8 && \
cd ../allsky && \
./make_links.sh && \
python allsky_init.py && \
python allsky_run.py && \
python compare-to-reference.py