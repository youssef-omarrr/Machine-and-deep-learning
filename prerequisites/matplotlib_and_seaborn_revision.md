# 🎨 **Matplotlib & Seaborn Extended Cheat Sheet**

### 📌 **Step 1: Import Libraries**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
```
- **matplotlib.pyplot**: For creating static, animated, or interactive plots.
- **seaborn**: For high-level statistical visualizations that integrate with Matplotlib.

---

# 🖼️ **Matplotlib: Basic & Advanced Visualization**

## 🔹 **1. Line Plot**  
```python
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, label='sin(x)', color='blue', linestyle='--', linewidth=2)
plt.xlabel("X-axis Label")
plt.ylabel("Y-axis Label")
plt.title("Sine Wave")
plt.legend()
plt.grid(True)
plt.show()
```
✅ **Key Functions:**  
- `plt.plot()`: Creates a line plot.  
- `label`: Adds a label for legend.  
- `color`: Specifies the line color.  
- `linestyle`: Changes the style (e.g., `--`, `-.`, `:`).  
- `linewidth`: Adjusts thickness.  
- `plt.xlabel()`, `plt.ylabel()`, `plt.title()`: Adds labels and title.  
- `plt.legend()`: Displays the legend.  
- `plt.grid(True)`: Shows the grid.  

---

## 🔹 **2. Scatter Plot**  
```python
x = np.random.rand(50)
y = np.random.rand(50)

plt.scatter(x, y, color='red', marker='o', alpha=0.6, s=100)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Scatter Plot")
plt.show()
```
✅ **Key Functions:**  
- `plt.scatter()`: Creates a scatter plot.  
- `marker='o'`: Specifies marker shape.  
- `alpha=0.6`: Adjusts transparency.  
- `s=100`: Sets marker size.  

---

## 🔹 **3. Bar Chart**  
```python
categories = ['A', 'B', 'C', 'D']
values = [10, 25, 15, 30]

plt.bar(categories, values, color='purple', alpha=0.7)
plt.xlabel("Categories")
plt.ylabel("Values")
plt.title("Bar Chart")
plt.show()
```
✅ **Key Functions:**  
- `plt.bar()`: Creates a bar chart.  
- `alpha=0.7`: Adjusts transparency.  

---

## 🔹 **4. Histogram**  
```python
data = np.random.randn(1000)

plt.hist(data, bins=30, color='green', edgecolor='black', alpha=0.7)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram")
plt.show()
```
✅ **Key Functions:**  
- `plt.hist()`: Creates a histogram.  
- `bins=30`: Defines number of bins.  
- `edgecolor='black'`: Adds borders to bars.  

---

## 🔹 **5. Pie Chart**  
```python
labels = ['Apple', 'Banana', 'Orange', 'Grapes']
sizes = [20, 30, 25, 25]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['red', 'yellow', 'orange', 'purple'])
plt.title("Fruit Distribution")
plt.show()
```
✅ **Key Functions:**  
- `plt.pie()`: Creates a pie chart.  
- `autopct='%1.1f%%'`: Displays percentage values.  

---

### 🔹 **6. Advanced Matplotlib Functions**

#### **6.1. Subplots and Figure Adjustments**
- **Creating Multiple Plots:**
  ```python
  fig, ax = plt.subplots(2, 2, figsize=(10, 8))
  ax[0, 0].plot(x, y, 'r-')
  ax[0, 1].scatter(x, y)
  ax[1, 0].bar(categories, values)
  ax[1, 1].hist(data, bins=30)
  plt.tight_layout()  # Adjusts subplot spacing automatically
  plt.show()
  ```
  - `plt.subplots()`: Creates a grid of subplots.
  - `figsize`: Sets the overall figure size.
  - `plt.tight_layout()`: Prevents overlapping content.

#### **6.2. Customizing Ticks & Labels**
- **Setting Custom Tick Values and Labels:**
  ```python
  plt.plot(x, y)
  plt.xticks(np.arange(0, 11, 2), ['0', '2', '4', '6', '8', '10'])
  plt.yticks(np.linspace(-1, 1, 5))
  plt.show()
  ```
  - `plt.xticks()` and `plt.yticks()`: Customize tick marks.

#### **6.3. Adding Annotations**
- **Annotate Specific Points:**
  ```python
  plt.plot(x, y)
  plt.annotate('Local Max', xy=(1.57, 1), xytext=(3, 1.2),
               arrowprops=dict(facecolor='black', shrink=0.05))
  plt.title("Annotated Plot")
  plt.show()
  ```
  - `plt.annotate()`: Adds text annotations with arrows.

#### **6.4. Saving Figures**
- **Save Your Plot to a File:**
  ```python
  plt.plot(x, y)
  plt.title("Save this Plot")
  plt.savefig("my_plot.png", dpi=300, bbox_inches='tight')
  plt.close()  # Closes the figure window
  ```
  - `plt.savefig()`: Exports your figure to an image file.
  - `dpi`: Controls the resolution.
  - `bbox_inches='tight'`: Minimizes whitespace.

#### **6.5. Changing Styles**
- **Using Predefined Styles:**
  ```python
  plt.style.use('ggplot')
  plt.plot(x, y)
  plt.title("Styled Plot")
  plt.show()
  ```
  - `plt.style.use()`: Apply a pre-built style (e.g., 'ggplot', 'seaborn', 'bmh').

---

# 🌟 **Seaborn: Advanced Visualization**  

## 🔹 **1. Line Plot**  
```python
tips = sns.load_dataset("tips")
sns.lineplot(x="size", y="total_bill", data=tips, marker="o")
plt.title("Line Plot of Bill by Size")
plt.show()
```
✅ **Key Functions:**  
- `sns.lineplot()`: Creates a line plot.  
- `data=tips`: Uses a dataset from Seaborn.  

---

## 🔹 **2. Scatter Plot with Regression Line**  
```python
sns.regplot(x="total_bill", y="tip", data=tips, scatter_kws={'alpha':0.5})
plt.title("Tip vs Total Bill")
plt.show()
```
✅ **Key Functions:**  
- `sns.regplot()`: Creates a scatter plot with a regression line.  
- `scatter_kws={'alpha':0.5}`: Adjusts transparency.  

---

## 🔹 **3. Bar Chart**  
```python
sns.barplot(x="day", y="total_bill", data=tips, palette="Blues")
plt.title("Average Bill per Day")
plt.show()
```
✅ **Key Functions:**  
- `sns.barplot()`: Creates a bar chart.  
- `palette="Blues"`: Changes color scheme.  

---

## 🔹 **4. Histogram (Distribution Plot)**  
```python
sns.histplot(tips["total_bill"], bins=20, kde=True, color="green")
plt.title("Distribution of Total Bills")
plt.show()
```
✅ **Key Functions:**  
- `sns.histplot()`: Creates a histogram.  
- `kde=True`: Adds a smooth density curve.  

---

## 🔹 **5. Box Plot (Outliers Detection)**  
```python
sns.boxplot(x="day", y="total_bill", data=tips, palette="Set2")
plt.title("Box Plot of Total Bill by Day")
plt.show()
```
✅ **Key Functions:**  
- `sns.boxplot()`: Creates a box plot for outlier detection.  

---

## 🔹 **6. Heatmap (Correlation Matrix)**  
```python
corr = tips.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
```
✅ **Key Functions:**  
- `sns.heatmap()`: Creates a heatmap.  
- `annot=True`: Displays correlation values.  

---

### 🔹 **7. Additional Seaborn Functions**

#### **7.1. Violin Plot**
- **Visualize the Distribution with Density Estimation:**
  ```python
  sns.violinplot(x="day", y="total_bill", data=tips, palette="muted")
  plt.title("Violin Plot of Total Bill by Day")
  plt.show()
  ```
  - Combines boxplot and KDE to show distribution shape.

#### **7.2. Pair Plot**
- **Quickly Visualize Pairwise Relationships:**
  ```python
  sns.pairplot(tips, hue="time", palette="Set1")
  plt.suptitle("Pair Plot of Tips Dataset", y=1.02)
  plt.show()
  ```
  - `sns.pairplot()`: Creates a grid of scatter plots and histograms for each pair of variables.
  
#### **7.3. Joint Plot**
- **Examine the Relationship Between Two Variables with Marginal Distributions:**
  ```python
  sns.jointplot(x="total_bill", y="tip", data=tips, kind="scatter", color="m")
  plt.suptitle("Joint Plot of Total Bill vs. Tip", y=1.02)
  plt.show()
  ```
  - `sns.jointplot()`: Combines a scatter plot with histograms/density plots on the margins.

#### **7.4. Count Plot**
- **Display the Count of Observations in Each Categorical Bin:**
  ```python
  sns.countplot(x="day", data=tips, palette="pastel")
  plt.title("Count of Observations per Day")
  plt.show()
  ```
  - `sns.countplot()`: Ideal for visualizing frequencies in categorical data.

#### **7.5. Cat Plot**
- **Combine Categorical Plots:**
  ```python
  sns.catplot(x="day", y="total_bill", hue="sex", data=tips, kind="swarm", palette="deep")
  plt.title("Cat Plot: Total Bill by Day and Sex")
  plt.show()
  ```
  - `sns.catplot()`: Flexible function that can create several types of categorical plots (swarm, box, violin, etc.).

#### **7.6. FacetGrid**
- **Create a Grid of Plots Based on a Categorical Variable:**
  ```python
  g = sns.FacetGrid(tips, col="time", height=4, aspect=1)
  g.map(plt.hist, "total_bill", bins=20, color="teal")
  g.add_legend()
  plt.show()
  ```
  - `sns.FacetGrid()`: Useful for plotting the same type of plot across subsets of your data.

#### **7.7. Customizing Themes & Contexts**
- **Set Style and Context Globally:**
  ```python
  sns.set_style("whitegrid")  # Options: white, dark, whitegrid, darkgrid, ticks
  sns.set_context("talk")      # Options: paper, notebook, talk, poster
  ```
  - Tailor the appearance of your plots to suit presentations or publications.

#### **7.8. Regression Plot Variants**
- **Using lmplot for Faceted Regression Plots:**
  ```python
  sns.lmplot(x="total_bill", y="tip", data=tips, hue="smoker", col="day", aspect=0.8)
  plt.suptitle("lmplot: Tip vs Total Bill Faceted by Day and Smoker Status", y=1.02)
  plt.show()
  ```
  - `sns.lmplot()`: Integrates regression plotting with facetting capabilities.

---

# 🎯 **Matplotlib vs. Seaborn: When to Use?**

| **Feature**           | **Matplotlib**           | **Seaborn**                               |
| --------------------- | ------------------------ | ----------------------------------------- |
| **Basic Plots**       | ✅ Yes                    | ✅ Yes                                     |
| **Styling**           | 🔶 Manual customization   | ✅ Automatic with themes                   |
| **Statistical Plots** | ❌ Requires manual coding | ✅ Built-in functions (box, violin, etc.)  |
| **Customization**     | ✅ Very High              | 🔶 Somewhat limited (but often enough)     |
| **Plot Facetting**    | ❌ Requires extra work    | ✅ Seamless with `FacetGrid` and `catplot` |

- **Matplotlib:**  
  - Full control for detailed, custom visualizations.
  - Ideal for building interactive dashboards and fine-tuning every aspect of your plot.

- **Seaborn:**  
  - Fast and beautiful statistical visualizations.
  - Great for exploratory data analysis with minimal code.

---
---

# Below is a step-by-step guide on how to integrate both Matplotlib and Seaborn in a Jupyter Notebook:

### 1. **Set Up Your Notebook**

- **Enable Inline Plotting:**  
  At the top of your notebook, run the magic command to display plots inline:
  ```python
  %matplotlib inline
  ```
  - **Purpose:** This is an IPython “magic” command that tells Jupyter to display Matplotlib plots directly in the notebook cells instead of opening them in a separate window.
  - **Details:**  
    - It sets up the backend to render static images.
    - Helps maintain an interactive workflow without leaving the notebook.
    - Alternative magics include `%matplotlib notebook` (for interactive plots) and `%matplotlib widget`.
  
- **Import Libraries:**  
  Import both Matplotlib and Seaborn along with any other necessary libraries:
  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns
  import numpy as np
  import pandas as pd
  ```

---

### 2. **Set a Seaborn Style**

- **Configure Seaborn:**  
  Seaborn integrates nicely with Matplotlib. Set a default style to improve the aesthetics of your plots:
  ```python
  sns.set(style="whitegrid")
  ```
  - **Purpose:** Applies Seaborn’s built-in theme to all plots.
  - **Details:**  
    - The `style` parameter controls the background of plots (options include `"white"`, `"dark"`, `"whitegrid"`, `"darkgrid"`, and `"ticks"`).
    - A `"whitegrid"` style adds a subtle grid that helps in reading values, which is especially useful for statistical plots.
    - This function influences Matplotlib’s settings, so subsequent plots (even those created solely with Matplotlib) inherit these styles.

---

### 3. **Create a Figure and Axes with Matplotlib**

- **Define a Figure/Axis:**  
  ```python
  fig, ax = plt.subplots(figsize=(8, 6))
  ```
  - **Purpose:** Creates a new figure (`fig`) and a set of subplots (`ax`).
  - **Details:**  
    - `plt.subplots()` is a versatile function that returns a tuple containing the figure object and an array (or single instance) of axes objects.
    - The `figsize` parameter specifies the width and height of the figure in inches.
    - The returned `ax` is used to control individual plot properties like titles, labels, and data display.

---

### 4. **Plot with Seaborn on the Matplotlib Axes**

- **Use Seaborn Plotting Functions:**  
  ```python
  # Example: Creating a box plot with Seaborn on a Matplotlib axis
  tips = sns.load_dataset("tips")
  sns.boxplot(x="day", y="total_bill", data=tips, ax=ax, palette="Set2")
  
  # Customize with Matplotlib functions
  ax.set_title("Total Bill Distribution by Day")
  ax.set_xlabel("Day of the Week")
  ax.set_ylabel("Total Bill")
  plt.show()
  ```

  - **Purpose:** Creates a box plot to visualize the distribution of `total_bill` across different `day` categories.
  - **Details:**  
    - `x` and `y` define the categorical and numerical variables, respectively.
    - `data=tips` specifies the DataFrame containing the data.
    - `ax=ax` tells Seaborn to plot on the previously created Matplotlib axis, allowing you to mix in Matplotlib customizations.
    - `palette="Set2"` selects a specific color palette to style the boxes.

---

### 5. **Mixing Matplotlib and Seaborn Commands**

- **Add Additional Matplotlib Customizations:**  
  ```python
  # Create a histogram using Seaborn's styling, then add a vertical line with Matplotlib
  fig, ax = plt.subplots(figsize=(8, 6))
  sns.histplot(tips["total_bill"], bins=20, kde=True, ax=ax, color="green")
  
  # Use Matplotlib to annotate the plot
  ax.axvline(x=tips["total_bill"].mean(), color="red", linestyle="--", label="Mean")
  ax.legend()
  ax.set_title("Histogram of Total Bills with Mean Line")
  plt.show()
  ```
  - **Purpose:** Draws a histogram of `total_bill` values.
  - **Details:**  
    - `bins=20` divides the data into 20 intervals.
    - `kde=True` overlays a kernel density estimate (smooth curve) to show the data distribution.
    - `color="green"` sets the color of the bars.
    - Using the `ax` parameter integrates this Seaborn plot into an existing Matplotlib subplot.
  
### `ax.axvline(x=tips["total_bill"].mean(), color="red", linestyle="--", label="Mean")`
- **Purpose:** Draws a vertical line at the mean value of `total_bill`.
- **Details:**  
  - `axvline` draws a vertical line across the axes at the specified `x` coordinate.
  - `color="red"` and `linestyle="--"` customize the appearance of the line.
  - `label="Mean"` adds a label that can be displayed in the legend.
  - This function is useful for highlighting key metrics within a plot.
  
### `ax.legend()`
- **Purpose:** Displays a legend on the plot.
- **Details:**  
  - The legend explains what each plotted element represents (e.g., the vertical line for the mean).
  - It automatically detects labeled elements (like the one from `axvline`).

### `ax.set_title("Histogram of Total Bills with Mean Line")`
- **Purpose:** Sets the title of the subplot.
- **Details:**  
  - Helps provide context for the plot.
  - Can be customized with font size, color, etc., if needed.

### `plt.show()`
- **Purpose:** Renders and displays the figure.
- **Details:**  
  - This function triggers the drawing of all active figures.
  - Essential when running scripts outside interactive environments; in Jupyter, plots are often rendered automatically after the cell executes.

---

### 6. **Using FacetGrid for Complex Layouts**

- **Combine Multiple Seaborn Plots:**  
  ```python
  g = sns.FacetGrid(tips, col="time", height=4, aspect=1)
  g.map(plt.hist, "total_bill", bins=20, color="teal")
  g.add_legend()
  
  # Adjust overall figure settings with Matplotlib
  plt.subplots_adjust(top=0.85)
  g.fig.suptitle("Total Bill Distribution by Time")
  plt.show()
  ```

### `g = sns.FacetGrid(tips, col="time", height=4, aspect=1)`
- **Purpose:** Creates a grid of subplots (facets) based on a categorical variable.
- **Details:**  
  - `col="time"` splits the data by the `time` column, creating a separate subplot for each unique value.
  - `height=4` sets the height (in inches) of each facet.
  - `aspect=1` controls the width-to-height ratio.
  - This grid makes it easier to compare distributions or trends across subsets of your data.

### `g.map(plt.hist, "total_bill", bins=20, color="teal")`
- **Purpose:** Maps a plotting function (here, `plt.hist`) onto each facet.
- **Details:**  
  - Applies the histogram function to the `total_bill` column in each facet.
  - Ensures each subplot receives the same formatting and parameters (e.g., number of bins and color).

### `g.add_legend()`
- **Purpose:** Adds a legend to the FacetGrid.
- **Details:**  
  - Useful when your mapped plots include color coding or multiple elements that need explanation.

### `plt.subplots_adjust(top=0.85)`
- **Purpose:** Adjusts the spacing of the subplot layout.
- **Details:**  
  - Specifically, it adjusts the top margin of the figure.
  - Useful to prevent overlapping titles or annotations when using multiple subplots or FacetGrid layouts.
  
### `plt.savefig("combined_plot.png", dpi=300, bbox_inches="tight")`
- **Purpose:** Saves the current figure to a file.
- **Details:**  
  - `"combined_plot.png"` is the file name.
  - `dpi=300` specifies a high resolution (dots per inch) for clear image quality.
  - `bbox_inches="tight"` trims extra whitespace around the figure, ensuring the saved image is neatly cropped.

### `plt.close()`
- **Purpose:** Closes the current figure.
- **Details:**  
  - Helps free up memory, especially when generating many plots in a loop or a large notebook.
  - Prevents figures from displaying automatically if you’re preparing multiple plots before a final display.

---

## 7. Adding Annotations

### `plt.annotate('Local Max', xy=(1.57, 1), xytext=(3, 1.2), arrowprops=dict(facecolor='black', shrink=0.05))`
- **Purpose:** Adds a text annotation with an arrow to highlight a specific point in the plot.
- **Details:**  
  - `xy=(1.57, 1)` defines the point to annotate.
  - `xytext=(3, 1.2)` sets the location of the annotation text.
  - `arrowprops` is a dictionary that customizes the arrow (e.g., `facecolor`, `shrink` factor).
  - This is particularly useful for highlighting features like peaks or outliers.

---

## Final Summary

When using both Matplotlib and Seaborn together:
- **Jupyter Setup:** Use `%matplotlib inline` for inline plot rendering.
- **Imports:** Bring in Matplotlib, Seaborn, NumPy, and Pandas.
- **Seaborn Styling:** Set a global style (like `"whitegrid"`) to enhance your plots.
- **Figure Creation:** Use `plt.subplots()` to create customizable figures and axes.
- **Plotting:** Utilize Seaborn’s functions (like `sns.boxplot`, `sns.histplot`) with the `ax` parameter to integrate with Matplotlib’s axes.
- **Customization:** Use Matplotlib functions (`ax.set_title()`, `ax.axvline()`, etc.) for detailed adjustments.
- **Advanced Layouts:** Use functions like `plt.subplots_adjust()` and `sns.FacetGrid` to manage multi-plot layouts.
- **Annotations and Saving:** Enhance plots with annotations and save them using `plt.savefig()`.

These detailed explanations provide you with the reasoning behind each function, ensuring that you can customize and troubleshoot your plots effectively. This integration gives you the best of both libraries—Seaborn’s beautiful statistical plots and Matplotlib’s fine-grained control.

---
