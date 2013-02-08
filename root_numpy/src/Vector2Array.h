#ifndef __VECTOR2ARRAY_H
#define __VECTOR2ARRAY_H
#include <vector>

// Workaround cython lack of template and pointer deref
template<typename T> class Vector2Array
{
    public:
        inline T* convert(std::vector<T>* v)
        {
            return v->size() > 0 ? &((*v)[0]) : NULL;
        }
};
#endif
