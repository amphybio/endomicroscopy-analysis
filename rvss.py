# @String source
# @String output
# @String xml
from register_virtual_stack import Register_Virtual_Stack_MT

print('\nStarting RVSS...')

# reference image
reference_name = "frame0000.png"

# shrinkage option (false)
use_shrinking_constraint = 0

# Parameters documentation
# https://javadoc.scijava.org/Fiji/register_virtual_stack/Register_Virtual_Stack_MT.Param.html
p = Register_Virtual_Stack_MT.Param()

# Initial sigma of each Scale Octave
p.sift.initialSigma = 1.6
# Steps per Scale Octave
p.sift.steps = 3
# The "minimal image size":
p.sift.minOctaveSize = 64
# The "maximum image size":
p.sift.maxOctaveSize = 1024


# Feature descriptor size.
p.sift.fdSize = 8
# Feature descriptor orientation bins How many bins per local histogram
p.sift.fdBins = 8
# Closest/next neighbor distance ratio
p.rod = 0.92

# Maximal allowed alignment error in pixels
p.maxEpsilon = 25.0
# The "inlier ratio":
p.minInlierRatio = 0.05

# Implemented transformation models for choice 0=TRANSLATION, 1=RIGID,
# 2=SIMILARITY, 3=AFFINE
p.featuresModelIndex = 1
# Implemented transformation models for choice 0=TRANSLATION, 1=RIGID,
# 2=SIMILARITY, 3=AFFINE, 4=ELASTIC, 5=MOVING_LEAST_SQUARES
p.registrationModelIndex = 0

# Interpolate
p.interpolate = True

Register_Virtual_Stack_MT.exec(source, output, xml,
                               reference_name, p, use_shrinking_constraint)
print('RVSS finished: {0}'.format(output))
