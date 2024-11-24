#CC = g++
UNAME := $(shell uname)
ifeq ($(UNAME),Darwin)
	CC = g++-14
else
	CC = g++
endif
LIBS = 
CFLAGS =-O2 -g -Wall -c -fmessage-length=0 -std=c++17
OPENMP = -fopenmp

RM := rm -rf

TARGET  = "OpenMP"

INCLUDE = -I../src

SUBDIRS  	 := $(wildcard ../) $(wildcard ../*/)
CPP_SRCS 	 := $(wildcard ../*/*.cpp) $(wildcard ../*/*/*.cpp) 

OBJS     := $(patsubst ../%.cpp, ./%.o, $(CPP_SRCS))
CPP_DEPS := $(patsubst ../%.cpp, ./%.d, $(CPP_SRCS))


# All Target
all: PSO

# Tool invocations
PSO: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	$(CC) -o $(TARGET) $(OPENMP) $(ARCH) $(OBJS) $(LIBS) 
	@echo 'Finished building target: $@'
	@echo ' '

src/%.o: ../src/%.cpp
	@mkdir -p $(@D)
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	$(CC) $(INCLUDE) $(CFLAGS) $(LPSOLVE_FLAGS) $(OPENMP) $(ARCH) $(LIBS) -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(OBJS)$(CPP_DEPS)$(CPP_DEPS) $(TARGET)
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY: