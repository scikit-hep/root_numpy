import numpy as np
cimport numpy as np

# TODO: use memcpy magic to get rid of weak typing assumption in copying array.
# The lines that looks like:
# ret[i_ret][i_land] = data[i_data][i_source][right_good_index]
# can be optimized since we know exactly how many bytes to copy
# and where to copy it to be careful of objects though you will
# need to INCREF it

cpdef _blockwise_inner_join(data, left, fk, right,
                            force_repeat, fk_name):
    # foreign key is given by array of scalar not array of array
    scalar_mode = fk.dtype != 'O' 

    # determine fk_name to be fk1 fk2 .... 
    # whichever is the first one that doesn't collide
    if fk_name is None:
        i_fk_name = 0
        fk_name = 'fk'
        while fk_name in left or fk_name in right:
            i_fk_name += 1
            fk_name = 'fk%d' % i_fk_name
    
    force_repeat = [] if force_repeat is None else force_repeat
    
    if scalar_mode:
        # auto repeat everything on the left in scalar mode
        # not really repeat since there would be exactly one copy
        force_repeat += left 

    repeat_columns = [c for c in left if data.dtype[c] != 'O' or c in force_repeat]
    cdef np.ndarray[np.int_t] repeat_indices = \
        np.array([data.dtype.names.index(x) for x in repeat_columns], np.int) 
    
    stretch_columns = [c for c in left if c not in repeat_columns]
    cdef np.ndarray[np.int_t] stretch_indices = \
        np.array([data.dtype.names.index(x) for x in stretch_columns], np.int)
    
    cdef np.ndarray[np.int_t] right_indices = \
        np.array([data.dtype.names.index(x) for x in right], np.int)
    
    # making new dtype
    new_dtype = []
    for c in left: # preserve order
        if c in repeat_columns:
            new_dtype.append((c, data.dtype[c]))
        elif c in stretch_columns:
            new_dtype.append((c, data[c][0].dtype))
    for c in right: # preserve order
        new_dtype.append((c, data[c][0].dtype))
    
    new_dtype.append((fk_name, fk[0].dtype))
    ret = None
    
    if scalar_mode: # scalar key mode
        ret = _scalar_fk_inner_join(
            data, right, fk, fk_name, new_dtype, 
            repeat_columns, stretch_columns,
            repeat_indices, stretch_indices, right_indices)
    else: # vector key mode
        ret = _vector_fk_inner_join(
            data, right, fk, fk_name, new_dtype, 
            repeat_columns, stretch_columns,
            repeat_indices, stretch_indices, right_indices)
    return ret


cdef _vector_fk_inner_join(np.ndarray data, right, np.ndarray fk,
                           fk_name,
                           new_dtype, 
                           repeat_columns, stretch_columns,
                           np.ndarray[np.int_t] repeat_indices,
                           np.ndarray[np.int_t] stretch_indices,
                           np.ndarray[np.int_t] right_indices):
    cdef long ndata = len(data)
    cdef np.ndarray first_right = data[right[0]]
    cdef np.ndarray good_fk_index = np.empty(ndata, 'O')
    cdef long nresult = 0
    cdef long i_data = 0
    cdef long max_fks
    cdef np.ndarray[np.int_t] good_index
    
    for i_data from 0 <= i_data < ndata:
        max_fks = len(first_right[i_data])
        fks = fk[i_data]
        good_index = np.flatnonzero((fks >= 0) & (fks < max_fks))
        nresult += len(good_index)
        good_fk_index[i_data] = good_index
    
    cdef np.ndarray ret = np.empty(nresult, new_dtype)
    
    # find where each of repeat/stretch/right lands
    cdef np.ndarray[np.int_t, ndim=1] repeat_result_indices = \
        np.array([ret.dtype.names.index(x) for x in repeat_columns], np.int)
    cdef np.ndarray[np.int_t, ndim=1] stretch_result_indices = \
        np.array([ret.dtype.names.index(x) for x in stretch_columns], np.int)
    cdef np.ndarray[np.int_t, ndim=1] right_result_indices = \
        np.array([ret.dtype.names.index(x) for x in right], np.int)

    cdef int fk_result_index = ret.dtype.names.index(fk_name)
    cdef long nrepeat = len(repeat_indices)
    cdef long nstretch = len(stretch_indices)
    cdef long nright = len(right_indices)
    cdef long left_good_index = 0
    cdef long right_good_index = 0
    cdef long i_land = 0
    cdef long i_source = 0
    cdef long i_repeat = 0
    cdef long i_stretch = 0
    cdef long i_right = 0
    cdef long i_ret = 0
    cdef long i_fk = 0
    cdef long this_n_good_fk = 0
    cdef np.ndarray[np.int_t] tmp_good_fk_index
    cdef np.ndarray tmp_fk

    for i_data from 0 <= i_data < ndata:
        tmp_good_fk_index = good_fk_index[i_data]
        tmp_fk = fk[i_data]
        this_n_good_fk = len(tmp_good_fk_index)
        for i_fk from 0 <= i_fk < this_n_good_fk:
            left_good_index = tmp_good_fk_index[i_fk]
            right_good_index = tmp_fk[left_good_index]
            for i_repeat from 0 <= i_repeat < nrepeat:
                i_land = repeat_result_indices[i_repeat]
                i_source = repeat_indices[i_repeat]
                ret[i_ret][i_land] = data[i_data][i_source] # TODO: make this faster
            for i_stretch from 0 <= i_stretch < nstretch:
                i_land = stretch_result_indices[i_stretch]
                i_source = stretch_indices[i_stretch]
                ret[i_ret][i_land] = data[i_data][i_source][left_good_index] # TODO: make this faster
            for i_right from 0 <= i_right < nright:
                i_land = right_result_indices[i_right]
                i_source = right_indices[i_right]
                ret[i_ret][i_land] = data[i_data][i_source][right_good_index] # TODO: make this faster
            ret[i_ret][fk_result_index] = right_good_index
            i_ret += 1
    return ret


cdef _scalar_fk_inner_join(np.ndarray data, right, np.ndarray fk,
                           fk_name, new_dtype, 
                           repeat_columns, stretch_columns,
                           np.ndarray[np.int_t] repeat_indices, 
                           np.ndarray[np.int_t] stretch_indices, 
                           np.ndarray[np.int_t] right_indices):
    cdef long ndata = len(data)
    cdef np.ndarray first_right = data[right[0]]
    cdef np.ndarray[np.int8_t, ndim=1] fk_index_good = np.empty(ndata, np.int8)
    cdef long max_fks
    cdef int fks = 0
    cdef long nresult = 0

    for i_data from 0 <= i_data < ndata:
        max_fks = len(first_right[i_data])
        fks = fk[i_data]
        fk_index_good[i_data] = (fks >= 0) and (fks < max_fks)
    
    nresult = np.count_nonzero(fk_index_good)
    ret = np.empty(nresult, new_dtype)
    
    # find where each of repeat/stretch/right lands
    cdef np.ndarray[np.int_t] repeat_result_indices = \
        np.array([ret.dtype.names.index(x) for x in repeat_columns], np.int)
    cdef np.ndarray[np.int_t] stretch_result_indices = \
        np.array([ret.dtype.names.index(x) for x in stretch_columns], np.int)
    cdef np.ndarray[np.int_t] right_result_indices = \
        np.array([ret.dtype.names.index(x) for x in right], np.int)
    cdef int fk_result_index = ret.dtype.names.index(fk_name)
    
    cdef long nrepeat = len(repeat_indices)
    cdef long nright = len(right_indices)
    cdef long i_repeat = 0
    cdef long i_right = 0
    cdef long i_land = 0
    cdef long i_source = 0
    cdef long i_ret = 0
    cdef long right_good_index=0

    for i_data from 0 <= i_data < ndata:
        if fk_index_good[i_data]:
            right_good_index = fk[i_data]
            for i_repeat from 0 <= i_repeat < nrepeat:
                i_land = repeat_result_indices[i_repeat]
                i_source = repeat_indices[i_repeat]
                ret[i_ret][i_land] = data[i_data][i_source]
            for i_right from 0 <= i_right < nright:
                i_land = right_result_indices[i_right]
                i_source = right_indices[i_right]
                ret[i_ret][i_land] = data[i_data][i_source][right_good_index]
            ret[i_ret][fk_result_index] = right_good_index
            i_ret += 1
    return ret
