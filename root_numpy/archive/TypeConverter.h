#include <vector>
#include <string>
#include <vector>
#include <map>
#include "Column.h"
using namespace std;

class ArrayKernel{
public:
	virtual void makeStructure(const vector<Column*> cols)=0;
	virtual void write(Column* col)=0;
	virtual bool convertible(const string& rtname)=0;
};

class ConverterPlugin{
public:
	virtual int convert(Column* col, void* buffer)=0;
};

class ArrayConverter{
	virtual int convert(Column* col, ConverterPlugin& cvp, void* buffer)=0;
}

class Converters{
public:
	map<string,ConverterPlugin> conv;

	virtual ArrayConverter* getFixedArrayConverter();
	virtual ArrayConverter* getVariableArrayConverter();

	virtual void add(const string& rtname,ConverterPlugin& conp){
		conv.insert(make_pair(rtname,conp))
	}

	virtual bool convertible(const string& rtname){
		return conv.find(rtname) != conv.end();
	}

	virtual int convert(Column* col, void* buffer){
		if(col->coltype==Column::SINGLE){

		}
		else if(col->coltype==Column::FIXED){

		}else if(col->coltype==Column::){

		}
	}	


};


