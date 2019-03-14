# Assignment 5

Welcome to your assignment 5! In this assignment you will use CNNs to predict on a modified version of the MNIST handwritten digits dataset. For this assignment we use two additional python packages which you must install IN ORDER:

First, activate your virtual environment
```
.mais-env\Scripts\activate
```

Then pip install the following:
```
pip install tensorflow
pip install keras
```

## 1. Setting up the environment

The assignments are set up as Jupyter Notebooks with cells that you have to fill in.

### Jupyter Notebooks
Jupyter notebooks are an interactive way to run python scripts. In a regular python script, you run the whole file at once. With notebooks, you can split the file into blocks called "cells" which can be run individually. Jupyter notebooks are widely used in the data science community so you should get used to them as early as possible!

### Python version
Python 3.7.2 was used for this assignment. However, any python 3 version should work. If your computer is 64-bit, ensure that you are running the 64-bit version of python. You can do this by opening a terminal and typing `python`. It will start up python in your terminal and display a message like:

```
Python 3.7.2 (tags/v3.7.2:9a3ffc0492, Dec 23 2018, 23:09:28) [MSC v.1916 64 bit (AMD64)] on win32 Type "help", "copyright", "credits" or "license" for more information.
```

Type `quit()` to exit the interpreter.

### Virtual environments
When starting a new python project, it is best practice to create a virtual environment to work in. A virtual environment is a python installation that is local to your project and can have its own packages that do not affect your global python installation. Imagine using the same python installation and set of packages for all your projects. This is a recipe for disaster as there could be a lot of conflicts! For example, different projects may require different versions of a package. We will set up a virtual environment for this and all future assignments. Open your terminal and do the following from your assignment folder:

1. Create a virtual environment called mais-env:
```
python -m venv .mais-env
```

2. Activate the environment so we are using the fresh, new python environment
```
.mais-env\Scripts\activate
```

3. Pip is pythons package manager used to install new packages. We have a local pip version in the virtual environment. Update it just in case it is not already updated:
```
python -m pip install --upgrade pip
```

4. requirements.txt specifies all packages used in the python project. This is a lightweight way of transferring projects from one user to another (eg. me transferring this assignment to you!) rather than transferring the whole virtual environment. You can then use the requirements.txt file to install all the dependencies in your newly created environment:
```
pip install -r requirements.txt
```

5. Now that we have our environment set up, how do we tell jupyter to use our environment? For some background, Jupyter notebooks support many languages and one of them is Python. A jupyter kernel is the backend used to run the actual code. Jupyter can be thought of as the "front-end" which connects to the IPython kernel "back-end" to run the code. To add a new kernel to jupyter, do the following:
```
python -m ipykernel install --user --name .mais-env --display-name "Python (MAIS-202)"
``` 

6. Open the notebook!
```
jupyter notebook
```

7. Click on ASsigment 1 and ensure the kernel is Python (MAIS-202). Otherwise, switch the kernel from Kernel > Change Kernel

#### Optional
To list all jupyter kernels, run `jupyter kernelspec list`. To remove kernels, run `jupyter kernelspec remove name-of-env`

## Submission

Create a pull request to the repository by doing the following:

1. Create a new branch:
```
git checkout -b firstname-lastname
```

For example
```
git checkout -b nabil-chowdhury
```

2. Commit your changes. Commit ONLY the "Assignment 1.ipynb" and the exported .py file
```
git add <file1> <file2>
git commit -m "your message"
```

For example
```
git add "Assignment 1.ipynb" "A1.py"
git commit -am "Finished assignment 1"
```

3. Push your changes. This actually pushes your branch to github.com
```
git push origin firstname-lastname
```

4. Create a pull request on github https://help.github.com/articles/creating-a-pull-request/

## Grading

We will provide feedback on your assignments in the pull requests as comments.
