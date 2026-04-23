## how to process data (pre processing)

run

```
python3 data_preprocessing
```

- NOTE: please edit the name of json files in the data_processings.py acc to whatever file name u are using


## how to run GAE file

```
cd cuda_spmm
python3 setup.py install
cd ..
python3 stage1_updated.py
```

## identifying labelled anomalies

```
python3 label_data.py
```
(check input data file paths, modify if needed)


## generating anomaly labels

```
python3 generate_labels.py
```

## running GAT (stage 2)

NOTE: make sure you have added `warp_gat_kernel.cu` inside cuda_spmm (if doing it now, run `python3 setup.py install` inside it again)

```
python3 stage2.py
```
