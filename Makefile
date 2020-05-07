MAKERULESDIR = ./trimesh2
DESTDIR = ./trimesh2/bin.$(UNAME)
INCLUDES = -I./trimesh2/include -Iinclude
LIBDIR = -L./trimesh2/lib.$(UNAME) -Llib.$(UNAME)

include $(MAKERULESDIR)/Makerules
CHOLMODLIBS = -lcholmod -lamd -lcolamd -lccolamd -lcamd -llapack -lblas
OPTSOURCES =	mesh_opt.cc

OPTFILES = $(addprefix $(OBJDIR)/,$(OPTSOURCES:.cc=.o))
OFILES = $(OPTFILES) 

OPTPROG = $(addsuffix $(EXE), $(addprefix $(DESTDIR)/, $(OPTSOURCES:.cc=)))
PROGS = $(OPTPROG)

default: $(PROGS)
	cp $(DESTDIR)/mesh_opt .


LIBS += -ltrimesh $(CHOLMODLIBS)

$(OPTPROG) : $(DESTDIR)/%$(EXE) : $(OBJDIR)/%.o
	$(LINK)

$(PROGS) : ./trimesh2/lib.$(UNAME)/libtrimesh.a


clean :
	-rm -f $(OFILES) $(OBJDIR)/Makedepend $(OBJDIR)/*.d
	-rm -rf $(OBJDIR)/ii_files
	-rmdir $(OBJDIR)

spotless : clean
	-rm -f $(PROGS)
	-rmdir $(DESTDIR)