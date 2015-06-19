#ifndef __COLUMN_H_
#define __COLUMN_H_

#include <TLeaf.h>
#include <TTreeFormula.h>
#include <string>


class Column
{
    public:

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


class FormulaColumn: public Column
{
    public:

    FormulaColumn(std::string _name, TTreeFormula* _formula)
    {
        name = _name;
        formula = _formula;
        type = "Double_t";
        value = new double[1];
    }

    ~FormulaColumn()
    {
        delete[] value;
    }

    int GetLen()
    {
        return 1;
    }

    int GetCountLen()
    {
        return 1;
    }

    int GetSize()
    {
        return sizeof(double) * GetLen();
    }

    void* GetValuePointer()
    {
        // required, as in TTreePlayer
        formula->GetNdata();
        value[0] = formula->EvalInstance(0);
        return value;
    }

    const char* GetTypeName()
    {
        return "double";
    }

    TTreeFormula* formula;
    double* value;
};


class BranchColumn: public Column
{
    public:

    BranchColumn(std::string& _name, TLeaf* _leaf)
    {
        name = _name;
        leaf = _leaf;
        type = leaf->GetTypeName();
    }

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
        // get len of this block (in unit of element)
        return leaf->GetLen();
    }

    int GetCountLen()
    {
        // get count leaf value
        TLeaf* count_leaf = leaf->GetLeafCount();
        if (count_leaf != NULL)
        {
            return int(count_leaf->GetValue());
        }
        return 1;
    }

    int GetSize()
    {
        // get size of this block in bytes
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
