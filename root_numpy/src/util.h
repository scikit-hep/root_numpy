#ifndef __UTIL_H_
#define __UTIL_H_
#include <typeinfo>
#include <cstdio>
#include <cstdarg>
#include <string>
#include <iostream>
#include <vector>

// Missing string printf
// This is safe and convenient but not exactly efficient
inline std::string format(const char* fmt, ...)
{
    int size = 512;
    char* buffer = 0;
    buffer = new char[size];
    va_list vl;
    va_start(vl, fmt);
    int nsize = vsnprintf(buffer, size, fmt, vl);
    if(size<=nsize)
    {
        // Delete buffer and try again
        delete buffer;
        buffer = 0;
        buffer = new char[nsize + 1]; // +1 for /0
        nsize = vsnprintf(buffer, size, fmt, vl);
    }
    std::string ret(buffer);
    va_end(vl);
    delete[] buffer;
    return ret;
}

// Workaround cython no ptr arithmetics rule
inline void* shift(void* v, int o)
{
    return (void*)((char*)v + o);
}

inline void printaddr(void* v)
{
    std::cout << std::hex << v << std::dec << std::endl;
}

template<typename T> class TypeName
{
    public:

    TypeName():
        name(typeid(T).name())
    {}

    const char* name;
};

// Workaround Cython's lack of template and pointer deref
template<typename T> class Vector2Array
{
    public:

    inline T* convert(std::vector<T>* v)
    {
        return v->size() > 0 ? &((*v)[0]) : NULL;
    }
};

#endif
