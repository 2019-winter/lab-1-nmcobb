---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Name(s)
**Miles Cobb**


**Instructions:** This is an individual assignment, but you may discuss your code with your neighbors.


# Python and NumPy

While other IDEs exist for Python development and for data science related activities, one of the most popular environments is Jupyter Notebooks.

This lab is not intended to teach you everything you will use in this course. Instead, it is designed to give you exposure to some critical components from NumPy that we will rely upon routinely.

## Exercise 0
Please read and reference the following as your progress through this course. 

* [What is the Jupyter Notebook?](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.ipynb#)
* [Notebook Tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)
* [Notebook Basics](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)

**In the space provided below, what are three things that still remain unclear or need further explanation?**


**YOUR ANSWER HERE**


## Exercises 1-7
For the following exercises please read the Python appendix in the Marsland textbook and answer problems A.1-A.7 in the space provided below.


## Exercise 1

```python
## YOUR SOLUTION HERE
## BEGIN SOLUTION
import numpy as np
a = np.ones((6,4)) * 2
a
## END SOLUTION
```

## Exercise 2

```python
## YOUR SOLUTION HERE
## BEGIN SOLUTION
b = np.ones((6,4))
np.fill_diagonal(b, 3)
b
## END SOLUTION
```

## Exercise 3


Using * does element by element multiplication. np.dot() attempts to multiply these matricies, which do not have dimensions that allow for matrix multiplication


## Exercise 4

```python
## YOUR SOLUTION HERE
## BEGIN SOLUTION
print(np.dot(a.transpose(),b))

print(np.dot(a,b.transpose()))

print("the first multiplies a 4x6 matrix by a 6x4 matrix, resulting in a 4x4.")
print("the second results in a 6x6 matrix as it multiplies a 6x4 by a 4x6")
## END SOLUTION

```

## Exercise 5

```python
## YOUR SOLUTION HERE
## BEGIN SOLUTION
def my_print(value):
    print(value)
    
my_print('testing')
## END SOLUTION
```

## Exercise 6

```python
## YOUR SOLUTION HERE
## BEGIN SOLUTION
def stats():
    arr1 = np.random.rand(3,3)
    arr2 = np.random.rand(3,3)
    print("random array 1", arr1)
    print("sum", np.sum(arr1))
    print("average", np.average(arr1))
    print("average column", np.average(arr1, 0))
    print("random array 2", arr2)
    print("sum", np.sum(arr1))
    print("average", np.average(arr1))
    print("average column", np.average(arr1, 0))
    
stats()
## END SOLUTION
```

## Exercise 7

```python
## YOUR SOLUTION HERE
## BEGIN SOLUTION
def loops(arr):
    count = 0 
    for row in arr:
        for col in row:
            if col == 1:
                count += 1
    return count

print("using loops:", loops(np.ones((5,4))))

def using_where(arr):
    return np.sum(np.where(arr == 1, 1, 0))

    

print("Using where:", using_where(np.ones((5,4))))
## END SOLUTION
```

## Excercises 8-???
While the Marsland book avoids using another popular package called Pandas, we will use it at times throughout this course. Please read and study [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) before proceeding to any of the exercises below.


## Exercise 8
Repeat exercise A.1 from Marsland, but create a Pandas DataFrame instead of a NumPy array.

```python
## YOUR SOLUTION HERE
## BEGIN SOLUTION
import pandas as pd
arr = np.ones((6,4))
df = pd.DataFrame(data=arr)
a = df * 2
a
## END SOLUTION
```

## Exercise 9
Repeat exercise A.2 using a DataFrame instead.

```python
## YOUR SOLUTION HERE
## BEGIN SOLUTION
b = np.ones((6,4))
np.fill_diagonal(b, 3)
b = pd.DataFrame(data=b)
b
## END SOLUTION
```

## Exercise 10
Repeat exercise A.3 using DataFrames instead.

```python
## YOUR SOLUTION HERE
## BEGIN SOLUTION
print(a * b) #multiplies element by element
try:
    np.dot(a, b)
except ValueError:
    print("does not work as multiplying a 6x4 by 6x4 is not valid")
## END SOLUTION
```

## Exercise 11
Repeat exercise A.7 using a dataframe.

```python
## YOUR SOLUTION HERE
## BEGIN SOLUTION
def loops(df):
    count = 0 
    for index, row in df.iterrows():
        for e in row:
            if e == 1:
                count += 1
        
    return count

def using_where(arr):
    return np.sum(np.where(arr == 1, 1, 0))


arr = pd.DataFrame(data = np.ones((5,3)))
print(arr)
print("using where", using_where(arr))
print("using loops", loops(arr))
## END SOLUTION
```

## Exercises 12-14
Now let's look at a real dataset, and talk about ``.loc``. For this exercise, we will use the popular Titanic dataset from Kaggle. Here is some sample code to read it into a dataframe.

```python
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df
```

Notice how we have nice headers and mixed datatypes? That is one of the reasons we might use Pandas. Please refresh your memory by looking at the 10 minutes to Pandas again, but then answer the following.


## Exercise 12
How do you select the ``name`` column without using .iloc?

```python
## YOUR SOLUTION HERE
## BEGIN SOLUTION
titanic_df['name']
## END SOLUTION
```

## Exercise 13
After setting the index to ``sex``, how do you select all passengers that are ``female``? And how many female passengers are there?

```python
## YOUR SOLUTION HERE
## BEGIN SOLUTION
titanic_df.set_index('sex',inplace=True)
women = titanic_df.loc['female']
women.size
## END SOLUTION
```

## Exercise 14
How do you reset the index?

```python
## YOUR SOLUTION HERE
## BEGIN SOLUTION
titanic_df.reset_index()
## END SOLUTION
```

```python

```

```python

```
