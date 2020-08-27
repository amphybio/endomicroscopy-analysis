# @String source
# @String output
# @String xml
from register_virtual_stack import Register_Virtual_Stack_MT

print('\nStarting RVSS...')

# reference image
reference_name = "frame000.png"

# shrinkage option (false)
use_shrinking_constraint = 0

p = Register_Virtual_Stack_MT.Param()
# The "maximum image size":
p.sift.maxOctaveSize = 1024
# The "inlier ratio":
p.minInlierRatio = 0.05

Register_Virtual_Stack_MT.exec(source, output, xml,
                               reference_name, p, use_shrinking_constraint)
print('RVSS finished: {0} \n'.format(output))
