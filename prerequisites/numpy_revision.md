## 🚀 **NumPy Cheat Sheet: Most Important Functions**  

### 📌 **Step 1: Import NumPy**
```python
import numpy as np
```
- `import numpy as np`: Imports NumPy, the core library for numerical computing.

---

## 🔹 **1. Creating NumPy Arrays**
```python
arr = np.array([1, 2, 3, 4, 5])
```
- Creates a **1D array** from a list.

```python
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
```
- Creates a **2D array (matrix)**.

```python
arr_zeros = np.zeros((3, 3))
```
- Creates a **3×3 matrix of zeros**.

```python
arr_ones = np.ones((2, 4))
```
- Creates a **2×4 matrix of ones**.

```python
arr_eye = np.eye(4)
```
- Creates a **4×4 identity matrix**.

```python
arr_random = np.random.rand(3, 3)
```
- Creates a **3×3 matrix with random values between 0 and 1**.

```python
arr_randint = np.random.randint(1, 100, (3, 3))
```
- Creates a **3×3 matrix with random integers between 1 and 100**.

```python
arr_linspace = np.linspace(0, 10, 5)
```
- Creates an array with **5 equally spaced values between 0 and 10**.

```python
arr_arange = np.arange(1, 10, 2)
```
- Creates an array from **1 to 10 with a step of 2**.

---

## 🔹 **2. Checking Array Properties**
```python
arr.shape
```
- Returns the **shape (rows, columns)** of the array.

```python
arr.size
```
- Returns the **total number of elements** in the array.

```python
arr.dtype
```
- Returns the **data type** of the array elements.

```python
arr.ndim
```
- Returns the **number of dimensions** of the array.

---

## 🔹 **3. Reshaping & Manipulating Arrays**
```python
arr.reshape(3, 2)
```
- Reshapes the array into **3 rows and 2 columns**.

```python
arr.flatten()
```
- Flattens a multi-dimensional array into **1D**.

```python
arr.T
```
- **Transposes** the array (rows become columns).

```python
arr.astype(int)
```
- Converts the array elements to **integers**.

---

## 🔹 **4. Indexing & Slicing**
```python
arr[0]  
```
- Accesses the **first element**.

```python
arr[1:4]  
```
- Slices elements **from index 1 to 3**.

```python
arr[:, 1]  
```
- Selects **all rows** of the **second column**.

```python
arr[1, :]  
```
- Selects **all columns** of the **second row**.

```python
arr[0:2, 1:3]  
```
- Selects **a subarray from row 0-1 and column 1-2**.

---

## 🔹 **5. Mathematical & Statistical Functions**
```python
np.sum(arr)
```
- Returns the **sum of all elements**.

```python
np.mean(arr)
```
- Returns the **mean (average) of the array**.

```python
np.median(arr)
```
- Returns the **median**.

```python
np.std(arr)
```
- Returns the **standard deviation**.

```python
np.var(arr)
```
- Returns the **variance**.

```python
np.min(arr)
```
- Returns the **minimum value**.

```python
np.max(arr)
```
- Returns the **maximum value**.

```python
np.argmax(arr)
```
- Returns the **index of the max value**.

```python
np.argmin(arr)
```
- Returns the **index of the min value**.

```python
np.unique(arr)
```
- Returns the **unique values** in the array.

```python
np.sort(arr)
```
- Sorts the array **in ascending order**.

---

## 🔹 **6. Element-wise Operations**
```python
arr1 + arr2  
```
- **Adds** corresponding elements.

```python
arr1 - arr2  
```
- **Subtracts** corresponding elements.

```python
arr1 * arr2  
```
- **Multiplies** corresponding elements.

```python
arr1 / arr2  
```
- **Divides** corresponding elements.

```python
np.exp(arr)
```
- Returns **exponential** of each element.

```python
np.sqrt(arr)
```
- Returns the **square root** of each element.

```python
np.log(arr)
```
- Returns the **natural log** of each element.

```python
np.abs(arr)
```
- Returns the **absolute value** of each element.

```python
np.round(arr, 2)
```
- Rounds values to **2 decimal places**.

---

## 🔹 **7. Boolean Masking & Filtering**
```python
arr[arr > 10]
```
- Returns **elements greater than 10**.

```python
arr[(arr > 5) & (arr < 15)]
```
- Returns **elements between 5 and 15**.

```python
np.where(arr > 10, 1, 0)
```
- Replaces values **greater than 10 with 1**, others with 0.

---

## 🔹 **8. Matrix Operations**
```python
np.dot(arr1, arr2)
```
- Computes the **dot product**.

```python
np.matmul(arr1, arr2)
```
- Computes **matrix multiplication**.

```python
np.linalg.inv(arr)
```
- Returns the **inverse** of a square matrix.

```python
np.linalg.det(arr)
```
- Computes the **determinant** of a matrix.

```python
np.linalg.eig(arr)
```
- Computes **eigenvalues & eigenvectors**.

---

## 🔹 **9. Saving & Loading Data**
```python
np.save("my_array.npy", arr)
```
- Saves an array to a **.npy** file.

```python
loaded_arr = np.load("my_array.npy")
```
- Loads a **.npy** file.

```python
np.savetxt("data.csv", arr, delimiter=",")
```
- Saves an array to a **CSV file**.

```python
arr_csv = np.loadtxt("data.csv", delimiter=",")
```
- Loads data from a **CSV file**.

---

### 🎯 **Summary Table**
| **Category**       | **Function**                      | **Description** |
|--------------------|--------------------------------|----------------|
| **Array Creation** | `np.array()`, `np.zeros()`, `np.ones()`, `np.eye()`, `np.linspace()`, `np.arange()` | Create different types of arrays |
| **Array Properties** | `shape`, `size`, `dtype`, `ndim` | Get array details |
| **Reshaping** | `reshape()`, `flatten()`, `T` | Modify array shape |
| **Indexing/Slicing** | `[start:end]`, `[:, col]`, `[row, :]` | Select elements |
| **Math Operations** | `sum()`, `mean()`, `median()`, `std()`, `var()` | Perform calculations |
| **Element-wise Ops** | `+`, `-`, `*`, `/`, `exp()`, `sqrt()`, `log()` | Apply functions |
| **Filtering** | `arr > 10`, `where()` | Boolean filtering |
| **Matrix Operations** | `dot()`, `matmul()`, `inv()`, `det()` | Perform linear algebra |

---

### ✅ **Final Notes**
- NumPy is **fast** due to **vectorized** operations.
- **Avoid loops**, use **NumPy functions** instead.
- Ideal for **data analysis, ML, and scientific computing**.

🔥 This cheat sheet covers the **most important NumPy functions**!