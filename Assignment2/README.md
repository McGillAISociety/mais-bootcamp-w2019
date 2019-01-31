# Assignment 2

Welcome to your second assignment! In the latest lecture, we went over linear and logistic regression - two of the most basic concepts in machine learning. This assignment will be focused on implementing some of the math formulas in actual code!

Just like your first assignments, please go through the python notebook labelled `assignment2.ipynb` and fill out the code. Remember - this assignment is for your own learning experience, try and attempt it yourself before asking/looking for help (Google is your best friend)! That being said, if Google is not able to help you, slack and office hours is that second group of friends that you can always count on!

Before starting, make sure you have a proper environment setup by following the instructions in Assignment 1 to install the required packages in the `requirements.txt`. Note that the required packages have not changed, so the same virtual environment can be used.

## Updating Your Branch

You should work off the same branch that you completed Assignment 1 on by doing the following:

1. Checkout master and pull the latest changes
```
git checkout master
git pull origin master
```

2. Checkout your own branch again and merge the latest changes from master
```
git checkout <YOUR BRANCH>
git merge master <YOUR BRANCH>
```

Note: Step 2 may require you to stage and commit your latest changes, in which case you should do that first.

## Submission

Submission will work in the same format as Assignment 1. Create a pull request to the repository by doing the following:

1. Commit your changes. Commit ONLY the "Assignment 2.ipynb" and the exported .py file
```
git add <file1> <file2>
git commit -m "your message"
```

For example
```
git add "Assignment 2.ipynb" "A2.py"
git commit -m "Finished assignment 2"
```

2. Push your changes. This actually pushes your branch to github.com
```
git push origin <YOUR BRANCH>
```

3. Create a pull request on github https://help.github.com/articles/creating-a-pull-request/

DO NOT MERGE THE PULL REQUEST!

## Grading

We will provide feedback on your assignments in the pull requests as comments.