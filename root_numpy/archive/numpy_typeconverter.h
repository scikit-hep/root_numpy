#include "TypeConverter.h"
#include <string>
#include <TLeaf.h>
#include "Column.h"
using namespace std;

    // root_typemap.insert(make_pair("Char_t",TypeInfo("i1",1,NPY_INT8)));
    // root_typemap.insert(make_pair("UChar_t",TypeInfo("u1",1,NPY_UINT8)));

    // root_typemap.insert(make_pair("Short_t",TypeInfo("i2",2,NPY_INT16)));
    // root_typemap.insert(make_pair("UShort_t",TypeInfo("u2",2,NPY_UINT16)));

    // root_typemap.insert(make_pair("Int_t",TypeInfo("i4",4,NPY_INT32)));
    // root_typemap.insert(make_pair("UInt_t",TypeInfo("u4",4,NPY_UINT32)));

    // root_typemap.insert(make_pair("Float_t",TypeInfo("f4",4,NPY_FLOAT32)));

    // root_typemap.insert(make_pair("Double_t",TypeInfo("f8",8,NPY_FLOAT64)));

    // root_typemap.insert(make_pair("Long64_t",TypeInfo("i8",8,NPY_INT64)));
    // root_typemap.insert(make_pair("ULong64_t",TypeInfo("u8",8,NPY_UINT64)));

    // root_typemap.insert(make_pair("Bool_t",TypeInfo("bool",1,NPY_BOOL)));

class Basic_Numpy_Converter: public ConverterPlugin{
	string rtname;
	string npytname;
	int size;
	NPY_TYPES npytype;
	Basic_Numpy_Converter(const string& rtname, const string* npytname, int size, NPY_Types npytype):
		rtname(rtname), npytname(npytname), size(size), npytype(npytype){}

	//this works for both fixed and single
	int convert(Column* col, void* buffer){
		TLeaf* leaf = col->leaf;
		assert(leaf!=NULL);
        void* src = leaf->GetValuePointer();
        assert(src!=NULL);
        ret = leaf->GetLenType()*leaf->GetLen();
        assert(ret>=0);
        memcpy(buffer,src,ret);
	}

}
