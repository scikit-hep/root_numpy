#ifndef __COLUMN_H_
#define __COLUMN_H_

#include <TLeaf.h>
#include <TTreeFormula.h>
#include <string>


class Column
{
    public:

    Column(std::string _name, std::string _type):
        name(_name),
        type(_type){}
    virtual ~Column() {}
    virtual int GetLen() = 0;
    virtual int GetCountLen() = 0;
    virtual int GetSize() = 0;
    virtual void* GetValuePointer() = 0;
    virtual const char* GetTypeName() = 0;

    // Column name
    std::string name;
    // Name of the ROOT type
    std::string type;
};


template <typename T>
class FormulaArrayColumn: public Column
{
    public:

    FormulaArrayColumn(std::string _name, std::string _type, TTreeFormula* _formula):
        Column(_name, _type),
        formula(_formula),
        value(NULL){}

    ~FormulaArrayColumn()
    {
        delete[] value;
    }

    const char* GetTypeName()
    {
        return type.c_str();
    }

    int GetLen()
    {
        return formula->GetNdata();
    }

    int GetCountLen()
    {
        return formula->GetNdata();
    }

    int GetSize()
    {
        return sizeof(T) * GetLen();
    }

    void* GetValuePointer()
    {
        delete[] value;
        value = new T[formula->GetNdata()];
        for (int i = 0; i < formula->GetNdata(); ++i)
        {
            value[i] = (T) formula->EvalInstance(i);
        }
        return value;
    }

    TTreeFormula* formula;
    T* value;
};


template <typename T>
class FormulaFixedArrayColumn: public FormulaArrayColumn<T>
{
    public:

    FormulaFixedArrayColumn(std::string _name, std::string _type, TTreeFormula* _formula):
        FormulaArrayColumn<T>(_name, _type, _formula)
    {
        length = _formula->GetNdata();
        this->value = new T[length];
    }

    int GetLen()
    {
        return length;
    }

    int GetCountLen()
    {
        return length;
    }

    void* GetValuePointer()
    {
        // Call to GetNdata() again required to update leaves
        for (int i = 0; i < this->formula->GetNdata(); ++i)
        {
            this->value[i] = (T) this->formula->EvalInstance(i);
        }
        return this->value;
    }

    int length;
};


template <typename T>
class FormulaColumn: public FormulaArrayColumn<T>
{
    public:

    FormulaColumn(std::string _name, std::string _type, TTreeFormula* _formula):
        FormulaArrayColumn<T>(_name, _type, _formula)
    {
        this->value = new T[1];
    }

    int GetLen()
    {
        return 1;
    }

    int GetCountLen()
    {
        return 1;
    }

    void* GetValuePointer()
    {
        this->formula->GetNdata(); // required, as in TTreePlayer
        this->value[0] = (T) this->formula->EvalInstance(0);
        return this->value;
    }
};


class BranchColumn: public Column
{
    public:

    BranchColumn(std::string _name, TLeaf* _leaf):
        Column(_name, _leaf->GetTypeName()),
        leaf(_leaf){}

    void SetLeaf(TLeaf* newleaf, bool check=false)
    {
        leaf = newleaf;
        if (check)
        {
            assert(leaf->GetTypeName() == type);
            // TODO: compare shape
        }
    }

    int GetLen()
    {
        // Get length of this block (number of elements)
        return leaf->GetLen();
    }

    int GetCountLen()
    {
        // Get count leaf value
        TLeaf* count_leaf = leaf->GetLeafCount();
        if (count_leaf != NULL)
        {
            return int(count_leaf->GetValue());
        }
        return 1;
    }

    int GetSize()
    {
        // Get size of this block in bytes
        return leaf->GetLenType() * leaf->GetLen();
    }

    void* GetValuePointer()
    {
        return leaf->GetValuePointer();
    }

    const char* GetTypeName()
    {
        return leaf->GetTypeName();
    }

    TLeaf* leaf;
};
#endif
