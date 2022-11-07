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