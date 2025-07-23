# Chapter 5 notes:

## [**SLIDES**](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/05_pytorch_going_modular.pdf)
## [**Book link**](https://www.learnpytorch.io/05_pytorch_going_modular/)
## [**Collab link part 1 (Cell mode)**](https://colab.research.google.com/github/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_cell_mode.ipynb/)
## [**Collab link part 2 (scripts mode)**](https://colab.research.google.com/github/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_script_mode.ipynb/)

## Cell magic commands:

In **Jupyter Notebooks**, `%%` is used for **cell magic commands**, which apply to an entire cell rather than a single line. These commands help with specialized tasks such as timing, running code in other languages, or configuring how the cell runs.

---

### 🧙‍♂️ Common `%%` Cell Magic Commands

| Magic                     | Description                                                    |
| ------------------------- | -------------------------------------------------------------- |
| `%%time`                  | Times how long the cell takes to run.                          |
| `%%timeit`                | Runs the cell multiple times to give average execution time.   |
| `%%writefile filename.py` | Writes the contents of the cell to an external file.           |
| `%%capture`               | Captures output of the cell (e.g., print statements or plots). |
| `%%bash`                  | Runs the cell content as a bash script.                        |
| `%%html`                  | Renders the cell contents as HTML.                             |
| `%%latex`                 | Renders LaTeX content.                                         |
| `%%javascript`            | Executes JavaScript in the Jupyter front end.                  |
| `%%python2` / `%%python3` | Runs the cell in Python 2 or Python 3 (if supported).          |

---

### ✅ Example Usage

```python
%%time
result = sum(range(10**6))
```

```python
%%writefile myscript.py
print("This is saved to a file")
```

```python
%%capture cap
print("This output is captured")
```

```python
%%bash
echo "Running a bash command"
```

---

### ❗Note:

* You can see all available magics using:

  ```python
  %lsmagic
  ```
* `%%` is for **cell magics**, while `%` is used for **line magics**.

--- 

## ✅ Brief on Google Style Docstrings

**Google style docstrings** are a standardized format for documenting Python code that is:

* Simple and readable
* Compatible with tools like Sphinx (with plugins)
* Widely used in professional Python codebases

---

### 📄 General Structure

```python
"""Summary line.

Extended description (optional).

Args:
    arg1 (type): Description.
    arg2 (type, optional): Description. Defaults to something.

Returns:
    type: Description of the return value.

Raises:
    ErrorType: Description of the error raised.
"""
```

---

### 🧪 Function Example

```python
def add_numbers(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a (int): First number.
        b (int): Second number.

    Returns:
        int: The sum of a and b.
    """
    return a + b
```

---

### 🧱 Class Example

```python
class Calculator:
    """A simple calculator class."""

    def __init__(self):
        """Initialize the calculator with no state."""
        self.history = []

    def multiply(self, x: float, y: float) -> float:
        """Multiply two numbers.

        Args:
            x (float): First number.
            y (float): Second number.

        Returns:
            float: The product of x and y.
        """
        result = x * y
        self.history.append(result)
        return result
```

---

### 📌 Notes

* No blank lines between sections like `Args`, `Returns`, etc.
* Optional arguments should be marked as `(type, optional)` with default values mentioned in the description.
* The return section can be omitted if the function returns `None`.

