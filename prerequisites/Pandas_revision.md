# This cheat sheet gives you a **detailed step-by-step guide** on handling CSV files using Pandas.

### 📌 **Step 1: Import Required Libraries**
```python
import numpy as np
import pandas as pd
```
- `import numpy as np`: Imports NumPy, a powerful library for numerical computing.
- `import pandas as pd`: Imports Pandas, which is essential for handling structured data.

---

### 📌 **Step 2: Read a CSV File**
```python
df = pd.read_csv("your_file.csv")  # Replace with your file path
```
- `pd.read_csv("your_file.csv")`: Reads a CSV file and loads it into a Pandas **DataFrame**, a table-like data structure.
- `header=None`: If your file doesn't have a header row, this treats the first row as data.
- `usecols=["column_name"]`: Reads only specific columns, improving efficiency.

Example:
```python
df = pd.read_csv("data.csv", usecols=["Name", "Age", "Salary"])
```

- To read without a header:
  ```python
  df = pd.read_csv("your_file.csv", header=None)
  ```

- To read only specific columns:
  ```python
  df = pd.read_csv("your_file.csv", usecols=["column_name"])
  ```

---

### 📌 **Step 3: Get General Information About the Data**
```python
df.info()  
```
- Displays column names, non-null counts, and data types.

```python
df.describe()
```
- Shows statistical summaries (count, mean, std, min, max, etc.) for numeric columns.

```python
df.shape
```
- Returns a tuple **(rows, columns)**, useful for checking dataset size.

```python
df.columns
```
- Lists all column names in the dataset.

```python
df.head()
```
- Displays the first 5 rows of the dataset (can specify a different number, e.g., `df.head(10)`).

```python
df.tail()
```
- Displays the last 5 rows of the dataset.

---

### 📌 **Step 4: Check for Null (Missing) Values**
```python
df.isnull().sum()
```
- Returns the count of missing values (`NaN`) per column.

```python
df.isna().sum()
```
- Same as `isnull()`, another way to check for missing data.

Example output:
```
Name       0
Age        2
Salary     3
dtype: int64
```
- This means **2 missing values in "Age"** and **3 missing values in "Salary"**.

---

### 📌 **Step 5: Handle Missing Values**

#### **Option 1: Drop Rows with Nulls**
```python
df.dropna(inplace=True)
```
- Removes all rows that contain at least one missing value.

#### **Option 2: Fill Nulls with a Specific Value**
```python
df.fillna(0, inplace=True)
```
- Replaces all missing values with `0`.

#### **Fill Nulls in a Specific Column**
```python
df["Age"].fillna(df["Age"].mean(), inplace=True)
```
- Replaces missing values in **"Age"** with the column's mean.

```python
df["Salary"].fillna(df["Salary"].median(), inplace=True)
```
- Replaces missing values in **"Salary"** with the median value.

```python
df["Name"].fillna(df["Name"].mode()[0], inplace=True)
```
- Fills missing values in **"Name"** with the most frequently occurring value (mode).

---

### 📌 **Step 6: Filter Top Data by a Certain Category**

#### **Get Top N Rows Based on a Column**
```python
df.nlargest(5, "Salary")  
```
- Returns the **top 5** rows with the highest salaries.

```python
df.nsmallest(5, "Salary")
```
- Returns the **bottom 5** rows with the lowest salaries.

#### **Filter Data by a Specific Category**
```python
df[df["Department"] == "IT"]
```
- Filters and returns only rows where the **"Department"** column is `"IT"`.

#### **Group by a Category and Get Top Values**
```python
df.groupby("Department")["Salary"].max()
```
- Groups by "Department" and returns the **maximum salary** in each department.

```python
df.groupby("Department")["Salary"].sum()
```
- Groups by "Department" and returns the **total salary** in each department.

```python
df.groupby("Department").mean()
```
- Groups by "Department" and returns the **average values** for numeric columns.

Example Output:
```
Department
HR       50000
IT       70000
Sales    45000
```
- The highest salary in **HR** is **$50,000**, in **IT** is **$70,000**, etc.

---

### 📌 **Bonus: Convert Data to NumPy Array**
```python
array_data = df.to_numpy()
```
- Converts the Pandas DataFrame into a **NumPy array**, useful for numerical processing.

---