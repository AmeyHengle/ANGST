#### v0.0.0 
- Desc: Original Train set (Stratified split 0.7 <> 0.3 from the annotation files)
- Silver Labels: None
- Silver Label technique: None
- Train size: 

#### v0.0.1 
- Desc: v0.0.0 + silver labels
- Silver Labels: fetched from unlabled_corpus.csv, topk=5, threshold=0.75
- Silver Label technique: SS
- Train size:
- Silver label distribution: 
    ```
    [1, 0]    460
    [0, 0]    338
    [1, 1]    302
    [0, 1]     31
    ```

#### v0.0.2
- Desc: v0.0.0 + silver labels
- Silver Labels: fetched from unlabled_corpus.csv, topk=3, threshold=0.75
- Silver Label technique: SS
- Train size:
- Silver label distribution: 
    ```
    [1, 0]    447
    [0, 0]    342
    [1, 1]    311
    [0, 1]     31
    ```

#### v0.0.3
- Desc: v0.0.0 + silver labels
- Silver Labels: fetched from unlabled_corpus.csv, topk=5, threshold=0.5
- Silver Label technique: SS
- Train size:
- Silver label distribution: 
    ```
    [0, 0]    4574
    [1, 0]    3112
    [1, 1]    2357
    [0, 1]     169
    ```

#### v0.0.4
- Desc: v0.0.0 + silver labels
- Silver Labels: fetched from unlabled_corpus.csv, topk=5, threshold=0.7
- Silver Label technique: SS
- Train size:
- Silver label distribution: 
    ```
    [1, 0]    1079
    [0, 0]     750
    [1, 1]     694
    [0, 1]      65
    ```

#### v0.0.5
- Desc: v0.0.0 + silver labels
- Silver Labels: fetched from unlabled_corpus.csv, topk=3
- Silver Label technique: BM25
- Train size:
- Silver label distribution: 
    ```
    [1, 1]    9763
    [1, 0]    8647
    [0, 0]    8635
    [0, 1]    1557
    ```

#### v0.0.5.1
- Desc: v0.0.0 + silver labels (sample_size=1000)
- Silver Labels: fetched from unlabled_corpus.csv, topk=3
- Silver Label technique: BM25
- Train size:
- Silver label distribution: 
    ```
    [0, 0]    1841
    [1, 0]    1679
    [1, 1]    1426
    [0, 1]    1064
    ```


#### v0.0.6
- Desc: v0.0.0 + silver labels
- Silver Labels: fetched from unlabled_corpus.csv, topk=1
- Silver Label technique: BM25
- Train size:
- Silver label distribution: 
    ```
    [0, 0]    10118
    [1, 1]     8755
    [1, 0]     8661
    [0, 1]     1068
    ```

#### v0.0.7
- Desc: v0.0.0 + silver labels
- Silver Labels: fetched from unlabled_corpus.csv, topk=5
- Silver Label technique: BM25
- Train size:
- Silver label distribution: 
    ```
    [1, 0]    9900
    [1, 1]    9889
    [0, 0]    8496
    [0, 1]     317
    ```

#### v0.0.8
- Desc: v0.0.0 + silver labels
- Silver Labels: fetched from unlabled_corpus.csv, threshold=0.5, topk=1
- Silver Label technique: SS
- Train size:
- Silver label distribution: 
    ```
    [0, 0]    4605
    [1, 0]    2989
    [1, 1]    2203
    [0, 1]     415
    ```

#### v0.1.0
- Desc: v0.0.0 + silver labels
- Silver Labels: fetched from unlabled_corpus.csv, topk=3, ss_threshold=0.7
- Silver Label technique: SS+BM25
- Train size:
- Silver label distribution: 
    ```
    [0, 0]    126
    [1, 1]    119
    [1, 0]     77
    [0, 1]     10
    ```

#### v0.1.1
- Desc: v0.0.0 + silver labels
- Silver Labels: fetched from unlabled_corpus.csv, topk=3, ss_threshold=0.5
- Silver Label technique: SS+BM25
- Train size:
- Silver label distribution: 
    ```
    [0, 0]    428
    [1, 1]    307
    [1, 0]    176
    [0, 1]     35
    ```

#### v0.1.2
- Desc: v0.0.0 + silver labels
- Silver Labels: fetched from unlabled_corpus.csv, topk=5, ss_threshold=0.5
- Silver Label technique: SS+BM25
- Train size:
- Silver label distribution: 
    ```
    [0, 0]    618
    [1, 1]    479
    [1, 0]    292
    [0, 1]     58
    ```

#### v0.1.3
- Desc: v0.0.0 + silver labels
- Silver Labels: fetched from unlabled_corpus.csv, topk=5, ss_threshold=0.35
- Silver Label technique: SS+BM25
- Train size:
- Silver label distribution: 
    ```
    [0, 0]    1076
    [1, 1]     636
    [1, 0]     416
    [0, 1]     110
    ```


#### v0.1.4
- Desc: v0.0.0 + silver labels
- Silver Labels: fetched from unlabled_corpus.csv, topk=2, ss_threshold=0.3
- Silver Label technique: DSM5
- Train size:
- Silver label distribution: 
    ```
    [1, 0]    1826
    [0, 1]     717
    [1, 1]     480
    ```

#### v0.1.5
- Desc: v0.0.0 + silver labels
- Silver Labels: fetched from unlabled_corpus.csv, topk=2, ss_threshold=0.15
- Silver Label technique: DSM5
- Train size:
- Silver label distribution: 
    ```
    [0, 1]    692
    [1, 0]    297
    [1, 1]    280
    ```

#### v1.0.0
- Desc: v0.0.0 + silver labels
- Silver Labels: fetched from unlabled_corpus.csv, topk=5, ss_threshold=0.7, GPT=True
- Silver Label technique: SS + GPT
- Train size:
- Silver label distribution: 
    ```
    [0, 0]    1076
    [1, 1]     636
    [1, 0]     416
    [0, 1]     110
    ```