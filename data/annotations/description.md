Abbreviations:
    a1: Madhumita
    a2: Jemima
    a3: Sneha
    mv: Majority Voting
    a1t: Atlest one true


Functions used:
    mv:
    - Get the majority vote label from a distribution array
    - lambda x: max(set(x), key=x.count)

    a1t:
    - Returns 1 if atleast one true label found in the annotations
    - lambda x: 1 if 1 in x else 0