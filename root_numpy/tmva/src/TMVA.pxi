
cdef extern from "2to3.h":
    pass 

cdef extern from "TMVA/Factory.h" namespace "TMVA":
    cdef cppclass Factory:
        void AddSignalTrainingEvent(vector[double]& event, double weight)
        void AddSignalTestEvent(vector[double]& event, double weight)
        void AddBackgroundTrainingEvent(vector[double]& event, double weight)
        void AddBackgroundTestEvent(vector[double]& event, double weight)

cdef extern from "TMVA/Reader.h" namespace "TMVA":
    cdef cppclass Reader:
        double EvaluateMVA(vector[double]& event, string name)
