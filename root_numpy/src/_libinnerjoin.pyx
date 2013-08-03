import numpy as np
cimport numpy as np

#TODO: use memcpy magic to get rid of weak typing assumption in copying array.
#The lines that looks like
#ret[i_ret][i_land] = data[i_data][i_source][right_good_index]
#can be optimized since we know exactly how many bytes to copy
#and where to copy it to
#be careful of objects though you will need to INCREF it

cpdef blockwise_inner_join(data, left, foreign_key, right,
                           force_repeat=None,
                           fk_name=None):
    """
    perform a blockwise inner join from names specified in left to right via 
    foreign_key left->foreign_key->right.
    
    Parameters
    ----------
    
    data : array
        full data set

    left : array
        array of left side column names

    foreign_key : array or string
        numpy array or string foreign_key column name
        This column can be either integer or array of int.
        if foreign_key is array of int column, left column will 
        be treated according to left column type:

        - Scalar columns or columns in force_repeat will be repeated

        - Array columns not in force_repeat will be assumed to the
          same length as foreign_key and will be strecthed by index 

    right : array
        array of right side column names
        These are array columns that each index foreign_key points to.
        These columns are assumed to have the same length.

    force_repeat : array
        array of left column names that 
        will be force to stretch even if it's an array(useful when
        you want to emulate multiple join)
    
    Examples
    --------

        >>> test_data = np.array([
        (1.0, np.array([11,12,13]), np.array([1,0,1]), 0, np.array([1,2,3])),
        (2.0, np.array([21,22,23]), np.array([-1,2,-1]), 1, np.array([31,32,33]))],
        dtype=[('sl', np.float), ('al', 'O'), ('fk', 'O'), ('s_fk', np.int), ('ar', 'O')])
        >>> blockwise_inner_join(test_data, ['sl', 'al'], test_data['fk'], ['ar'] )
        array([(1.0, 11, 2, 1), (1.0, 12, 1, 0), (1.0, 13, 2, 1), (2.0, 22, 33, 2)], 
        dtype=[('sl', '<f8'), ('al', '<i8'), ('ar', '<i8'), ('fk', '<i8')])
        >>> blockwise_inner_join(test_data, ['sl','al'], test_data['fk'], ['ar'], force_repeat=['al'])
        array([(1.0, [11, 12, 13], 2, 1), (1.0, [11, 12, 13], 1, 0),
        (1.0, [11, 12, 13], 2, 1), (2.0, [21, 22, 23], 33, 2)], 
        dtype=[('sl', '<f8'), ('al', '|O8'), ('ar', '<i8'), ('fk', '<i8')])

    """
    fk = foreign_key if not isinstance(foreign_key, basestring) else data[foreign_key]
    
    scalar_mode = fk.dtype != 'O'#foreign key is given by array of scalar not array of array
    

    #determine fk_name to be fk1 fk2 .... 
    #whichever the first one that doesn't collide
    if fk_name is None:
        i_fk_name = 1
        fk_name = 'fk%d'%i_fk_name
        while fk_name in left or fk_name in right:
            i_fk_name += 1
            fk_name = 'fk%d'%i_fk_name
    
    force_repeat = [] if force_repeat is None else force_repeat
    
    if scalar_mode:
        #auto repeat everything on the left in scalar mode
        #not really repeat since there would be exactly one copy
        force_repeat += left 

    repeat_columns = [c for c in left if data.dtype[c]!='O' or c in force_repeat]
    cdef np.ndarray[np.int_t] repeat_indices = \
        np.array(map(data.dtype.names.index, repeat_columns), np.int) 
    
    stretch_columns = [c for c in left if c not in repeat_columns]
    cdef np.ndarray[np.int_t] stretch_indices = \
        np.array(map(data.dtype.names.index, stretch_columns), np.int)
    
    cdef np.ndarray[np.int_t] right_indices = \
        np.array(map(data.dtype.names.index, right), np.int)
    
    #making new dtype
    new_dtype = []
    for c in left: #preserve order-ish
        if c in repeat_columns:
            new_dtype.append((c,data.dtype[c]))
        elif c in stretch_columns:
            new_dtype.append((c,data[c][0].dtype))
    for c in right: #preserve order_ish
        new_dtype.append((c,data[c][0].dtype))
    
    new_dtype.append((fk_name,fk[0].dtype))
    ret = None
    
    if scalar_mode: #scalar key mode
        ret = _scalar_fk_inner_join(data, right, fk, fk_name, new_dtype, 
                           repeat_columns, stretch_columns,
                           repeat_indices, stretch_indices, right_indices)
    else: #vector key mode
        ret = _vector_fk_inner_join(data, right, fk, fk_name, new_dtype, 
                           repeat_columns, stretch_columns,
                           repeat_indices, stretch_indices, right_indices)
    return ret


cdef _vector_fk_inner_join(np.ndarray data, right,  np.ndarray fk, fk_name,
                           new_dtype, 
                           repeat_columns, stretch_columns,
                           np.ndarray[np.int_t] repeat_indices,
                           np.ndarray[np.int_t] stretch_indices,
                           np.ndarray[np.int_t] right_indices):
    cdef int ndata = len(data)
    cdef np.ndarray first_right = data[right[0]]
    cdef np.ndarray good_fk_index = np.empty(ndata,'O')
    
    cdef int nresult = 0
    cdef int i_data = 0
    cdef int max_fks
    cdef np.ndarray[np.int_t] good_index
    
    for i_data in range(ndata):
        max_fks = len(first_right[i_data])
        fks = fk[i_data]
        good_index = np.flatnonzero((fks>=0) & (fks<max_fks))
        nresult += len(good_index)
        good_fk_index[i_data] = good_index
    
    cdef np.ndarray ret = np.empty(nresult, new_dtype)
    
    #find where each of repeat/stretch/right lands
    cdef np.ndarray[np.int_t, ndim=1] repeat_result_indices = \
            np.array(map(ret.dtype.names.index, repeat_columns), np.int)
    cdef np.ndarray[np.int_t, ndim=1] stretch_result_indices = \
            np.array(map(ret.dtype.names.index, stretch_columns), np.int)
    cdef np.ndarray[np.int_t, ndim=1] right_result_indices = \
            np.array(map(ret.dtype.names.index, right) , np.int)
    cdef int fk_result_index = ret.dtype.names.index(fk_name)
    
    cdef int nrepeat = len(repeat_indices)
    cdef int nstretch = len(stretch_indices)
    cdef int nright = len(right_indices)
    
    cdef int left_good_index = 0
    cdef int right_good_index = 0
    
    cdef int i_land = 0
    cdef int i_source = 0
    
    cdef int i_repeat = 0
    cdef int i_stretch = 0
    cdef int i_right = 0
    #let's do the real join
    cdef int i_ret = 0
    cdef int i_fk = 0
    
    cdef int this_n_good_fk = 0
    
    cdef np.ndarray[np.int_t] tmp_good_fk_index
    cdef np.ndarray tmp_fk
    for i_data in range(ndata):
        tmp_good_fk_index = good_fk_index[i_data]
        tmp_fk = fk[i_data]
        this_n_good_fk = len(tmp_good_fk_index)
        
        for i_fk in range(this_n_good_fk):
    
            left_good_index = tmp_good_fk_index[i_fk]
            right_good_index = tmp_fk[left_good_index]
            
            for i_repeat in range(nrepeat):
                i_land = repeat_result_indices[i_repeat]
                i_source = repeat_indices[i_repeat]
                ret[i_ret][i_land] = data[i_data][i_source] #<< make this faster
            
            for i_stretch in range(nstretch):
                i_land = stretch_result_indices[i_stretch]
                i_source = stretch_indices[i_stretch]
                ret[i_ret][i_land] = data[i_data][i_source][left_good_index] #<< make this faster
            
            for i_right in range(nright):
                i_land = right_result_indices[i_right]
                i_source = right_indices[i_right]
                ret[i_ret][i_land] = data[i_data][i_source][right_good_index] #<< make this faster
            
            ret[i_ret][fk_result_index] = right_good_index
            i_ret+=1
    return ret


cdef _scalar_fk_inner_join(np.ndarray data, right, np.ndarray fk,
                           fk_name, new_dtype, 
                           repeat_columns, stretch_columns,
                           np.ndarray[np.int_t] repeat_indices, 
                           np.ndarray[np.int_t] stretch_indices, 
                           np.ndarray[np.int_t] right_indices):
    cdef int ndata = len(data)
    cdef np.ndarray first_right = data[right[0]]
    cdef np.ndarray[np.int8_t, ndim=1] fk_index_good = np.empty(ndata,np.int8)
    cdef int fks = 0
    
    nresult = 0
    for i_data in range(ndata):
        max_fks = len(first_right[i_data])
        fks = fk[i_data]
        fk_index_good[i_data] = (fks>=0) and (fks<max_fks)
    
    nresult = np.count_nonzero(fk_index_good)
    
    ret = np.empty(nresult, new_dtype)
    
    #find where each of repeat/stretch/right lands
    cdef np.ndarray[np.int_t] repeat_result_indices = \
        np.array(map(ret.dtype.names.index, repeat_columns ), np.int)
    cdef np.ndarray[np.int_t] stretch_result_indices = \
        np.array(map(ret.dtype.names.index, stretch_columns ), np.int)
    cdef np.ndarray[np.int_t] right_result_indices = \
        np.array(map(ret.dtype.names.index, right ), np.int)
    cdef int fk_result_index = ret.dtype.names.index(fk_name)
    
    #let's do the real join
    i_ret = 0
    cdef int nrepeat = len(repeat_indices)
    cdef int nright = len(right_indices)
    
    cdef int i_repeat = 0
    cdef int i_right = 0
    cdef int i_land = 0
    cdef int i_source = 0
    cdef int right_good_index=0
    for i_data in range(ndata):
        if fk_index_good[i_data]:
            
            right_good_index = fk[i_data]
            
            for i_repeat in range(nrepeat):
                i_land = repeat_result_indices[i_repeat]
                i_source = repeat_indices[i_repeat]
                ret[i_ret][i_land] = data[i_data][i_source]
            
            for i_right in range(nright):
                i_land = right_result_indices[i_right]
                i_source = right_indices[i_right]
                ret[i_ret][i_land] = data[i_data][i_source][right_good_index]
            
            ret[i_ret][fk_result_index] = right_good_index
            i_ret+=1
    return ret
