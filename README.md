Code and supplementary material for: Stratified Sampling for Extreme Multi-label Data

# Instructions

### 1. Download and extract <strong>stratify_function</strong> folder
```
stratify_function
|--- stratify.py
|--- helper_funcs.py
```
### 2. Import function into python workspace

```python
import sys
sys.path.append('/absolute/path/to/folder/stratify_function')

from stratify import stratified_train_test_split
```

### 3. Use the function

```python
### Example usage
X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, target_test_size=0.2, random_state=42)
```

### Usage notes
- Works very similarly to test_train_split from scikit-learn
- X and y need to be lists.
- y needs to be a list of lists, where each inner list contains the label set for each document in X
- X can be a list of anything (lists, strings, integers, dicts, etc.). The contents of X is not used during partitioning.
- The generated test size can be different from the target test size. Please refer to paper for details.
- The contents of X_train and X_test will be of the same data type as X.
- The contents of y_train and y_test will be a list of lists, where each inner list contains the label set for each document in X_train and X_test respectively.