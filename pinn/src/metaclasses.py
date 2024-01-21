import types

    
class MetaParameters(type):
    def __new__(self, class_name, bases, attrs):
        Lname = 'Lx'
        Nname = 'Nx'
        Dname = 'Dx'
        for Lname, Nname, Dname in [['Lx', 'Nx', 'Dx'],
                                   ['Ly', 'Ny', 'Dy'],
                                   ['Lt', 'Nt', 'Dt']]:
            # Check existance of Lname, Dname and Nname
            isL = Lname in attrs
            isN = Nname in attrs
            isD = Dname in attrs
            if isL + isN + isD != 2:
                raise Exception(f"Exactly 2 of parameters {Lname}, {Nname}, {Dname} must me set")

            # Fill the missing variable
            if not isL: attrs[Lname] = attrs[Nname] * attrs[Dname]
            if not isN: attrs[Nname] = attrs[Lname] / attrs[Dname]
            if not isD: attrs[Dname] = attrs[Lname] / attrs[Nname]
                
            # Convert N to int
            attrs[Nname] = int(attrs[Nname])

        # run setup if exists
        cls = type(class_name, bases, attrs)           
        if 'setup' in attrs: cls.setup(cls)
        return cls
    
class MetaSetup(type):
     def __new__(self, class_name, bases, attrs):
            
        # run setup if exists
        cls = type(class_name, bases, attrs)
        if 'setup' in attrs: cls.setup(cls)
        return cls