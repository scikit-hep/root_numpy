# Makefile.arch can be in a number of different locations
ARCH_LOC_1 := $(wildcard $(shell root-config --prefix)/test/Makefile.arch)
ARCH_LOC_2 := $(wildcard $(shell root-config --prefix)/share/root/test/Makefile.arch)
ARCH_LOC_3 := $(wildcard $(shell root-config --prefix)/share/doc/root/test/Makefile.arch)
ARCH_LOC_4 := $(wildcard $(shell root-config --etcdir)/Makefile.arch)
ifneq ($(strip $(ARCH_LOC_1)),)
  $(info Using $(ARCH_LOC_1))
  include $(ARCH_LOC_1)
else
  ifneq ($(strip $(ARCH_LOC_2)),)
    $(info Using $(ARCH_LOC_2))
    include $(ARCH_LOC_2)
  else
    ifneq ($(strip $(ARCH_LOC_3)),)
      $(info Using $(ARCH_LOC_3))
      include $(ARCH_LOC_3)
    else
      ifneq ($(strip $(ARCH_LOC_4)),)
        $(info Using $(ARCH_LOC_4))
        include $(ARCH_LOC_4)
      else
        $(error Could not find Makefile.arch!)
      endif
    endif
  endif
endif 

# Name of the executable
EXE = generate

# List of all source files to build
CPPLIST = $(wildcard *.$(SrcSuf))

CXXFLAGS += -g

# The default is to create the executable
default: $(EXE)

# Rule to create the dictionaries
dicts.cxx: LinkDef.h
	@echo "Generating dictionaries"
	@rootcint -f $@ -c $(CXXFLAGS) -p $^

# Rule to create the executable
$(EXE): dicts.cxx $(CPPLIST)
	@echo "Making executable: $@"
	@rm -f $@
	@$(CXX) $^ $(CXXFLAGS) $(INCLUDES) $(LIBS) -o $@

# Rule to clean the directory
clean:
	@rm -f $(EXE)
	@rm -f dicts.cxx
	@rm -f dicts.h
