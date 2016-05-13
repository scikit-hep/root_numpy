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


class MultiFormulaColumn: public Column
{
    public:

    MultiFormulaColumn(std::string _name, TTreeFormula* _formula)
    {
        name = _name;
        formula = _formula;
        type = "Double_t";
        value = NULL;
    }

    ~MultiFormulaColumn()
    {
        delete[] value;
    }

    const char* GetTypeName()
    {
        return "double";
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
        return sizeof(double) * GetLen();
    }

    void* GetValuePointer()
    {
        delete[] value;
        value = new double[formula->GetNdata()];
        for (int i = 0; i < formula->GetNdata(); ++i)
        {
            value[i] = formula->EvalInstance(i);
        }
        return value;
    }

    TTreeFormula* formula;
    double* value;
};


class FormulaColumn: public MultiFormulaColumn
{
    public:

    FormulaColumn(std::string _name, TTreeFormula* _formula):
        MultiFormulaColumn(_name, _formula)
    {
        value = new double[1];
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
        formula->GetNdata(); // required, as in TTreePlayer
        value[0] = formula->EvalInstance(0);
        return value;
    }
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
