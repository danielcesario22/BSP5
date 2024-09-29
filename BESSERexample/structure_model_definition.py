
from besser.BUML.metamodel.structural import DomainModel, Class, Property, \
    PrimitiveDataType, Multiplicity, BinaryAssociation,Constraint,Method
from besser.BUML.metamodel.object import *
from RL_code_generator import RLGenerator

#############################################
#   DRL - structural model definition   #
#############################################

# DRL model class definition
drl_model: Class = Class (name="DRLmodel")

# Domain model definition
DRL : DomainModel = DomainModel(name="DRL", types={drl_model}, 
                                          associations={},
                                          constraints={})

