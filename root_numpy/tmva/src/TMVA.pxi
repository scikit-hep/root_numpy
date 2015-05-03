
cdef extern from "2to3.h":
    pass 

cdef extern from "TMVA/Types.h" namespace "TMVA":
    ctypedef enum ETreeType "TMVA::Types::ETreeType":
        kTraining "TMVA::Types::kTraining"
        kTesting "TMVA::Types::kTesting"

cdef extern from "TMVA/Event.h" namespace "TMVA":
    cdef cppclass Event:
        Event(vector[float]& features, unsigned int theclass)
        void SetVal(unsigned int ivar, float value)

cdef extern from "TMVA/Factory.h" namespace "TMVA":
    cdef cppclass Factory:
        void AddEvent(string& classname, ETreeType treetype, vector[double]& event, double weight)

cdef extern from "TMVA/MethodBase.h" namespace "TMVA":
    cdef cppclass MethodBase:
        double GetMvaValue()
        vector[float] GetMulticlassValues()
        vector[float] GetRegressionValues()
        Event* fTmpEvent
