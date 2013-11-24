#ifndef __COLUMN_H_
#define __COLUMN_H_

#include <TLeaf.h>
#include <TTreeFormula.h>
#include <string>


enum ColumnType
{
    SINGLE = 1,
    FIXED = 2,
    VARY = 3
};


class Column
{
    public:
        virtual ~Column() {}
        virtual int GetLen() = 0;
        virtual int GetSize() = 0;
        virtual void* GetValuePointer() = 0;
        virtual const char* GetTypeName() = 0;

        bool skipped;
        // single fixed vary?
        ColumnType coltype;
        // column name
        std::string colname;
        // useful in case of fixed element
        int countval;
        // name of the roottype
        std::string rttype;
};


class FormulaColumn: public Column
{
    public:

        FormulaColumn(std::string _colname, TTreeFormula* _formula)
        {
            colname = _colname;
            coltype = SINGLE;
            formula = _formula;
            rttype = "Double_t";
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


// This describes the structure of the tree
// Converter should take this and make the appropriate data structure
class BranchColumn: public Column
{
    public:

        static int find_coltype(TLeaf* leaf,
                                ColumnType& coltype,
                                int& countval)
        {
            // Check whether it's array if so of which type
            TLeaf* len_leaf = leaf->GetLeafCounter(countval);
            if (countval == 1)
            {
                if (len_leaf == 0)
                { // single element
                    coltype = SINGLE;
                }
                else
                { // variable length
                    coltype = VARY;
                }
            }
            else if (countval > 0)
            {
                // fixed multiple array
                coltype = FIXED;
            }
            else
            {
                // negative
                std::string msg("Unable to understand the structure of leaf ");
                msg += leaf->GetName();
                PyErr_SetString(PyExc_IOError, msg.c_str());
                return 0;
            }
            return 1;
        }

        static BranchColumn* build(TLeaf* leaf, const std::string& colname)
        {
            BranchColumn* ret = new BranchColumn();
            ret->leaf = leaf;
            ret->colname = colname;
            ret->skipped = false;
            if (!find_coltype(leaf, ret->coltype, ret->countval))
            {
                delete ret;
                return NULL;
            }
            ret->rttype = leaf->GetTypeName();
            return ret;
        }

        void SetLeaf(TLeaf* newleaf, bool check=false)
        {
            leaf = newleaf;
            if (check)
            {
                assert(leaf->GetTypeName() == rttype);
                int cv;
                ColumnType ct;
                if (find_coltype(leaf, ct, cv) == 0)
                    abort();
                if (ct != coltype)
                    abort();
                //if(ct==FIXED){assert(cv==countval);}
            }
        }

        int GetLen()
        {
            // get len of this block (in unit of element)
            return leaf->GetLen();
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
