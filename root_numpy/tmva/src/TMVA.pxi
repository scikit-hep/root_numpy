
cdef extern from "2to3.h":
    pass 

cdef extern from "TMVA/Types.h" namespace "TMVA":
    ctypedef enum EMVA "TMVA::Types::EMVA":
        kCuts "TMVA::Types::kCuts"

    ctypedef enum ETreeType "TMVA::Types::ETreeType":
        kTraining "TMVA::Types::kTraining"
        kTesting "TMVA::Types::kTesting"

    ctypedef enum EAnalysisType "TMVA::Types::EAnalysisType":
        kClassification "TMVA::Types::kClassification"
        kRegression "TMVA::Types::kRegression"
        kMulticlass "TMVA::Types::kMulticlass"

cdef extern from "TMVA/Event.h" namespace "TMVA":
    cdef cppclass Event:
        Event(vector[float]& features, unsigned int theclass)
        void SetVal(unsigned int ivar, float value)

cdef extern from "TMVA/DataSetInfo.h" namespace "TMVA":
    cdef cppclass DataSetInfo:
        unsigned int GetNClasses()
        unsigned int GetNVariables()
        unsigned int GetNTargets()
        vector[string] GetListOfVariables()

cdef extern from "TMVA/IMethod.h" namespace "TMVA":
    cdef cppclass IMethod:
        pass

cdef extern from "TMVA/MethodBase.h" namespace "TMVA":
    cdef cppclass MethodBase:
        EMVA GetMethodType()
        EAnalysisType GetAnalysisType()
        DataSetInfo DataInfo()
        unsigned int GetNVariables()
        unsigned int GetNTargets()
        double GetMvaValue()
        vector[float] GetMulticlassValues()
        vector[float] GetRegressionValues()
        Event* fTmpEvent

cdef extern from "TMVA/MethodCuts.h" namespace "TMVA":
    cdef cppclass MethodCuts:
        void SetTestSignalEfficiency(double eff)

cdef extern from "TMVA/Factory.h" namespace "TMVA":
    cdef cppclass Factory:
        void AddEvent(string& classname, ETreeType treetype, vector[double]& event, double weight)

cdef extern from "TMVA/Reader.h" namespace "TMVA":
    cdef cppclass Reader:
        IMethod* FindMVA(string name)
