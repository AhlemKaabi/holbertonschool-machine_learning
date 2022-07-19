# **Pandas**

<p align="center">
  <img src="./img/pandas_logo.jpg">
</p>

## **Learning Objectives**
* What is `pandas`?
* What is a `pd.DataFrame`? How do you create one?
	* `pd.DataFrame` is  `class pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)`
	* The primary pandas data structure.
	* Two-dimensional, size-mutable, potentially heterogeneous tabular data.
* What is a `pd.Series`? How do you create one?
	* `pd.Series` is `class pandas.Series(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False)`
	* One-dimensional ndarray with axis labels (including time series).
	* Example &#10549;
	```
	d = {'a': 1, 'b': 2, 'c': 3}

	ser = pd.Series(data=d, index=['a', 'b', 'c'])
	```
* How to load data from a file
	```
	df = pd.read_csv('file_name.csv')
	```
* How to perform indexing on a `pd.DataFrame`
* How to use hierarchical indexing with a `pd.DataFrame`
* How to slice a `pd.DataFrame`
	* using iloc: `DataFrame.iloc` or loc: `DataFrame.loc`
* How to reassign columns
	```
	# A base reassign (naming)
	df["column_name"] = "new_name"
	```
* How to sort a `pd.DataFrame`
	```
	# Sort by the values along either axis.
	DataFrame.sort_values(by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False, key=None)

	# Sort object by labels (along an axis).
	DataFrame.sort_index(axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, ignore_index=False, key=None)
	```
* How to use boolean logic with a `pd.DataFrame`
* How to merge/concatenate/join `pd.DataFrame`s
* How to get statistical information from a `pd.DataFrame`
	```
	df.describe()
	```
* How to visualize a `pd.DataFrame`
	* using matplotlib



## **Resources**
* [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
* [Complete Python Pandas Data Science Tutorial! (Reading CSV/Excel files, Sorting, Filtering, Groupby)](https://www.youtube.com/watch?v=vmEHCJofslg&ab_channel=KeithGalli)
* [Matplotlib Crash Course](https://www.youtube.com/watch?v=3Xc3CA655Y4)

