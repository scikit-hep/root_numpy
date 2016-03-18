
.. currentmodule:: root_numpy

==============
root_numpy FAQ
==============

This is a list of Frequently Asked Questions about root_numpy. Feel free to
suggest new entries!

How do I add a new branch to an existing tree?
----------------------------------------------

If your tree is large and the values of the new branch depend on values in
other branches, it is probably best to use ROOT directly and set up a loop that
fills a new branch entry-by-entry. Otherwise, you can simply use root_numpy's
:func:`array2tree` function and specify the existing tree you want the branch
added to with the ``tree`` argument. Please see the documentation, examples,
and notes for :func:`array2tree`. Note that even if the new branch will depend
on values in other branches, you can first convert the tree into a numpy array
(see :func:`tree2array`) and define your new structured numpy array based on
columns in this array before adding the new branch(es) with :func:`array2tree`.

How do I remove a branch from a tree?
-------------------------------------

You can use root_numpy to read an array from a tree with :func:`tree2array` and
request only the branches you want kept in the ``branches`` argument. Then
convert the array back into a new tree with :func:`array2tree`. But, note that
this involves reading the tree into memory all at once. For large trees, this
won't be efficient or might not even be possible if you don't have enough
memory.

Instead, just use ROOT directly. Deactivate the branches before calling
the tree's ``CloneTree()`` method. Only the activated branches are copied::

   from root_numpy.testdata import get_filepath
   from rootpy import asrootpy
   from rootpy.io import root_open

   rfile = root_open(get_filepath('single1.root'))
   tree = rfile['tree']
   print tree.branchnames  # prints ['n_int', 'f_float', 'd_double']

   # remove the branch named 'd_double'
   tree.SetBranchStatus('d_double', 0)

   # copy the tree into a new file
   with root_open('output.root', 'w'):
       newtree = asrootpy(tree.CloneTree())
       newtree.Write()
       print newtree.branchnames  # prints ['n_int', 'f_float']
